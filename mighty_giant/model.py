from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HybridConfig
from .mamba3.block import Mamba3Block
from .memory.fast_weights import LowRankFastWeightMemory
from .memory.mag import MAGInjection
from .memory.mac import MACFusionBlock
from .memory.surprise import MemoryUpdater
from .state import LowRankState, SSMState, SequenceState
from .swiglu import SwiGLU


@dataclass
class MightyGiantOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None
    state: SequenceState


class MightyGiantLM(nn.Module):
    """Mamba-3 backbone + Titans long-term memory hybrid language model.

    Architecture per the initial-thoughts.md design:
      - Interleaved Mamba3Block + SwiGLU layers (Llama pattern)
      - MAG gated memory injection at every layer (cheap path)
      - Sparse MAC fusion blocks every ``fusion_block_every`` layers (expensive path)
      - Low-rank fast-weight memory updated per segment via surprise
      - Persistent memory slots in MAC fusion blocks

    Supports staged training:
      - memory_enabled=False: pure Mamba-3 backbone (Stage 1)
      - memory_enabled=True: full hybrid (Stages 2-3)
    """

    def __init__(self, config: HybridConfig, *, memory_enabled: bool = True):
        super().__init__()
        self.config = config
        self.memory_enabled = memory_enabled

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Backbone: interleaved Mamba3Block + SwiGLU
        self.mamba_blocks = nn.ModuleList()
        self.ffn_blocks = nn.ModuleList()
        for i in range(config.n_layers):
            self.mamba_blocks.append(Mamba3Block(config, layer_idx=i))
            self.ffn_blocks.append(SwiGLU(config.d_model, config.ffn_hidden_resolved))

        # Memory components
        self.memory = LowRankFastWeightMemory(
            config.d_model, config.memory_depth, config.fast_weight_rank
        )
        self.memory_updater = MemoryUpdater(config.d_model)
        self.memory_query_proj = nn.Linear(config.d_model, config.d_model)

        # MAG injection at every layer
        self.mag_injections = nn.ModuleList(
            [MAGInjection(config.d_model) for _ in range(config.n_layers)]
        )

        # Sparse MAC fusion blocks
        self.fusion_layers = {}
        self.fusion_blocks = nn.ModuleDict()
        for i in range(config.n_layers):
            if config.is_fusion_layer(i):
                self.fusion_blocks[str(i)] = MACFusionBlock(config)
                self.fusion_layers[i] = str(i)

        self.final_norm = nn.RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        self._apply_residual_scaling()

        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self, module: nn.Module) -> None:
        # GPT-2/3 style: scale residual projections by 1/sqrt(2*n_layers)
        std = 0.02
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def _apply_residual_scaling(self) -> None:
        """Scale down residual-path projections to prevent signal explosion at init."""
        scale = 1.0 / (2 * self.config.n_layers) ** 0.5
        for block in self.mamba_blocks:
            with torch.no_grad():
                block.out_proj.weight.mul_(scale)
        for block in self.ffn_blocks:
            with torch.no_grad():
                block.w2.weight.mul_(scale)

    def init_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> SequenceState:
        ssm_states = []
        for block in self.mamba_blocks:
            ssm_states.append(
                SSMState(h=block.init_state(batch_size, device=device, dtype=dtype))
            )

        memory_state = self.memory.init_state(batch_size, device=device, dtype=dtype)
        return SequenceState(
            ssm_states=ssm_states,
            memory_state=memory_state,
            tokens_seen=0,
        )

    def _retrieve_memory_slots(
        self,
        query: torch.Tensor,
        state: SequenceState,
    ) -> torch.Tensor:
        """Retrieve multiple memory slots for MAC fusion.

        Args:
            query: [batch, d_model] segment query.
            state: current sequence state.

        Returns:
            [batch, retrieved_memory_slots, d_model]
        """
        n_slots = self.config.retrieved_memory_slots
        batch = query.size(0)
        d = query.size(-1)

        # Generate multiple queries by learned linear combinations
        # For simplicity, use the same query shifted by small learned offsets
        slots = []
        for i in range(n_slots):
            # Slight perturbation per slot to get diverse retrievals
            scale = 1.0 + 0.1 * (i - n_slots / 2)
            slot_query = query * scale
            slot = self.memory.forward_with_state(
                F.normalize(slot_query, dim=-1, eps=1e-6),
                state.memory_state,
            )
            slots.append(slot)

        return torch.stack(slots, dim=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        labels: torch.Tensor | None = None,
        state: SequenceState | None = None,
        reset_state: bool = False,
    ) -> MightyGiantOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq_len].")

        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        dtype = self.token_embedding.weight.dtype

        if state is None or reset_state:
            state = self.init_state(batch_size, device=device, dtype=dtype)
        elif not self.training:
            state = state.enable_grad()

        h = self.token_embedding(input_ids)

        segment_len = self.config.segment_len
        all_outputs = []

        for seg_start in range(0, seq_len, segment_len):
            seg_end = min(seg_start + segment_len, seq_len)
            segment = h[:, seg_start:seg_end, :]

            # Memory read (once per segment)
            if self.memory_enabled:
                seg_query = self.memory_query_proj(segment.mean(dim=1))
                memory_readout = self.memory.forward_with_state(
                    F.normalize(seg_query, dim=-1, eps=1e-6),
                    state.memory_state,
                )
                retrieved_slots = self._retrieve_memory_slots(seg_query, state)

            # Process through all layers
            new_ssm_states = []
            for i in range(self.config.n_layers):
                # Mamba-3 SSM block
                segment, new_ssm_h = self.mamba_blocks[i](
                    segment, state.ssm_states[i].h
                )
                new_ssm_states.append(SSMState(h=new_ssm_h))

                # MAG injection (cheap, every layer)
                if self.memory_enabled:
                    segment = self.mag_injections[i](segment, memory_readout)

                # SwiGLU FFN
                segment = self.ffn_blocks[i](segment)

                # Sparse MAC fusion (expensive, select layers)
                if self.memory_enabled and i in self.fusion_layers:
                    segment = self.fusion_blocks[self.fusion_layers[i]](
                        segment, retrieved_slots
                    )

            # Memory write (update with post-backbone hidden states)
            if self.memory_enabled:
                mem_state_for_update = state.memory_state.enable_grad()
                new_memory_state = self.memory_updater.update(
                    segment, self.memory, mem_state_for_update
                )
            else:
                new_memory_state = state.memory_state

            state = SequenceState(
                ssm_states=new_ssm_states,
                memory_state=new_memory_state,
                tokens_seen=state.tokens_seen + (seg_end - seg_start),
            )

            all_outputs.append(segment)

        hidden = torch.cat(all_outputs, dim=1)
        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            shifted_logits = logits[:, :-1, :].contiguous()
            shifted_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shifted_logits.view(-1, shifted_logits.size(-1)),
                shifted_labels.view(-1),
            )

        if not self.training:
            logits = logits.detach()
            state = state.detach()
            if loss is not None:
                loss = loss.detach()

        return MightyGiantOutput(logits=logits, loss=loss, state=state)
