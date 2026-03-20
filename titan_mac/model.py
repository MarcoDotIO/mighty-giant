from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig


def batched_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    out = torch.einsum("bti,boi->bto", x, weight)
    return out + bias[:, None, :]


def split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    batch, seq_len, width = x.shape
    head_dim = width // n_heads
    return x.view(batch, seq_len, n_heads, head_dim).transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    batch, n_heads, seq_len, head_dim = x.shape
    return x.transpose(1, 2).contiguous().view(batch, seq_len, n_heads * head_dim)


@dataclass
class FastWeightState:
    weights: list[torch.Tensor]
    biases: list[torch.Tensor]
    momentum_weights: list[torch.Tensor]
    momentum_biases: list[torch.Tensor]

    def detach(self) -> "FastWeightState":
        return FastWeightState(
            weights=[tensor.detach() for tensor in self.weights],
            biases=[tensor.detach() for tensor in self.biases],
            momentum_weights=[tensor.detach() for tensor in self.momentum_weights],
            momentum_biases=[tensor.detach() for tensor in self.momentum_biases],
        )

    def enable_grad(self) -> "FastWeightState":
        return FastWeightState(
            weights=[tensor.detach().requires_grad_(True) for tensor in self.weights],
            biases=[tensor.detach().requires_grad_(True) for tensor in self.biases],
            momentum_weights=[tensor.detach() for tensor in self.momentum_weights],
            momentum_biases=[tensor.detach() for tensor in self.momentum_biases],
        )


@dataclass
class ModelState:
    layer_states: list[FastWeightState]
    tokens_seen: int = 0

    def detach(self) -> "ModelState":
        return ModelState(
            layer_states=[state.detach() for state in self.layer_states],
            tokens_seen=self.tokens_seen,
        )

    def enable_grad(self) -> "ModelState":
        return ModelState(
            layer_states=[state.enable_grad() for state in self.layer_states],
            tokens_seen=self.tokens_seen,
        )


@dataclass
class TitansOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None
    state: ModelState


class DepthwiseSeparableProjection(nn.Module):
    def __init__(self, width: int, kernel_size: int):
        super().__init__()
        self.linear = nn.Linear(width, width)
        self.depthwise = nn.Conv1d(
            width,
            width,
            kernel_size=kernel_size,
            groups=width,
            padding=kernel_size - 1,
        )
        self.pointwise = nn.Conv1d(width, width, kernel_size=1)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.activation(self.linear(x))
        projected = projected.transpose(1, 2)
        projected = self.depthwise(projected)
        projected = self.pointwise(projected)
        projected = projected[..., : x.size(1)]
        return projected.transpose(1, 2)


class GatedFeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_size: int):
        super().__init__()
        self.up = nn.Linear(d_model, hidden_size)
        self.gate = nn.Linear(d_model, hidden_size)
        self.down = nn.Linear(hidden_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activated = F.silu(self.up(x))
        gated = torch.sigmoid(self.gate(x))
        return self.down(activated * gated)


class MACBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.pre_attn_norm = nn.LayerNorm(config.d_model)
        self.pre_ffn_norm = nn.LayerNorm(config.d_model)
        self.attn_q = DepthwiseSeparableProjection(config.d_model, config.conv_kernel_size)
        self.attn_k = DepthwiseSeparableProjection(config.d_model, config.conv_kernel_size)
        self.attn_v = DepthwiseSeparableProjection(config.d_model, config.conv_kernel_size)
        self.attn_gate = nn.Linear(config.d_model, config.d_model)
        self.attn_out = nn.Linear(config.d_model, config.d_model)
        self.ffn = GatedFeedForward(config.d_model, config.ffn_hidden)

        self.persistent_memory = nn.Parameter(
            torch.randn(config.persistent_tokens, config.d_model) * 0.02
        )
        self.prefix_position = nn.Parameter(
            torch.randn(config.prefix_tokens, config.d_model) * 0.02
        )

        self.memory_query_proj = nn.Linear(config.d_model, config.d_model)
        self.memory_key_proj = nn.Linear(config.d_model, config.d_model)
        self.memory_value_proj = nn.Linear(config.d_model, config.d_model)

        self.alpha_head = nn.Linear(config.d_model, 1)
        self.eta_head = nn.Linear(config.d_model, 1)
        self.theta_head = nn.Linear(config.d_model, 1)
        nn.init.constant_(self.theta_head.bias, -3.0)

        memory_dims = [config.d_model]
        for _ in range(config.memory_depth - 1):
            memory_dims.append(config.memory_hidden)
        memory_dims.append(config.d_model)
        self.memory_layers = nn.ModuleList(
            [
                nn.Linear(memory_dims[index], memory_dims[index + 1])
                for index in range(len(memory_dims) - 1)
            ]
        )

    def init_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> FastWeightState:
        weights = []
        biases = []
        momentum_weights = []
        momentum_biases = []
        for layer in self.memory_layers:
            layer_weight = (layer.weight * 0.01).to(device=device, dtype=dtype).unsqueeze(0).expand(
                batch_size, -1, -1
            )
            layer_bias = (layer.bias - layer.bias.detach()).to(device=device, dtype=dtype).unsqueeze(
                0
            ).expand(batch_size, -1)
            weights.append(layer_weight)
            biases.append(layer_bias)
            momentum_weights.append(torch.zeros_like(layer_weight))
            momentum_biases.append(torch.zeros_like(layer_bias))
        return FastWeightState(
            weights=weights,
            biases=biases,
            momentum_weights=momentum_weights,
            momentum_biases=momentum_biases,
        )

    def _memory_forward(self, x: torch.Tensor, state: FastWeightState) -> torch.Tensor:
        hidden = x
        for index, (weight, bias) in enumerate(zip(state.weights, state.biases)):
            hidden = batched_linear(hidden, weight, bias)
            if index < len(state.weights) - 1:
                hidden = F.silu(hidden)
        return hidden

    def _pool_queries(self, hidden_states: torch.Tensor) -> torch.Tensor:
        seq_len = hidden_states.size(1)
        if seq_len == self.config.memory_tokens:
            pooled = hidden_states
        else:
            positions = torch.linspace(
                0,
                seq_len - 1,
                steps=self.config.memory_tokens,
                device=hidden_states.device,
            )
            gather_index = positions.round().long().clamp(max=seq_len - 1)
            pooled = hidden_states.index_select(1, gather_index)
        queries = self.memory_query_proj(F.silu(pooled))
        return F.normalize(queries, dim=-1, eps=1e-6)

    def _make_prefix(self, retrieved: torch.Tensor) -> torch.Tensor:
        batch_size = retrieved.size(0)
        persistent = self.persistent_memory.to(
            device=retrieved.device, dtype=retrieved.dtype
        ).unsqueeze(0).expand(batch_size, -1, -1)
        prefix_position = self.prefix_position.to(
            device=retrieved.device, dtype=retrieved.dtype
        ).unsqueeze(0).expand(batch_size, -1, -1)
        persistent = persistent + prefix_position[:, : self.config.persistent_tokens, :]
        retrieved = retrieved + prefix_position[:, self.config.persistent_tokens :, :]
        return torch.cat([persistent, retrieved], dim=1)

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        query = F.normalize(self.attn_q(x), dim=-1, eps=1e-6)
        key = F.normalize(self.attn_k(x), dim=-1, eps=1e-6)
        value = self.attn_v(x)
        query = split_heads(query, self.config.n_heads)
        key = split_heads(key, self.config.n_heads)
        value = split_heads(value, self.config.n_heads)
        attn = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )
        attn = merge_heads(attn)
        gated = attn * torch.sigmoid(self.attn_gate(attn))
        return self.attn_out(gated)

    def _update_memory(self, segment_states: torch.Tensor, state: FastWeightState) -> FastWeightState:
        pooled = segment_states.mean(dim=1)
        alpha = torch.sigmoid(self.alpha_head(pooled))
        eta = torch.sigmoid(self.eta_head(pooled))
        theta = F.softplus(self.theta_head(pooled)) + 1e-6

        keys = F.normalize(self.memory_key_proj(F.silu(segment_states)), dim=-1, eps=1e-6)
        values = self.memory_value_proj(F.silu(segment_states))
        predicted = self._memory_forward(keys, state)
        associative_loss = F.mse_loss(predicted, values, reduction="mean")

        fast_params = state.weights + state.biases
        grads = torch.autograd.grad(
            associative_loss,
            fast_params,
            create_graph=self.training,
            retain_graph=self.training,
        )

        # Clip inner gradients to prevent fast-weight explosion.
        # Compute global norm across all grad tensors, then scale uniformly.
        total_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads))
        clip_coef = (1.0 / (total_norm + 1e-6)).clamp(max=1.0)
        grads = tuple(g * clip_coef for g in grads)

        alpha_w = alpha.unsqueeze(-1)
        eta_w = eta.unsqueeze(-1)
        theta_w = theta.unsqueeze(-1)
        alpha_b = alpha
        eta_b = eta
        theta_b = theta

        new_weights = []
        new_biases = []
        new_momentum_weights = []
        new_momentum_biases = []
        weight_count = len(state.weights)
        for index, weight in enumerate(state.weights):
            grad_weight = grads[index]
            grad_bias = grads[index + weight_count]
            momentum_weight = eta_w * state.momentum_weights[index] - theta_w * grad_weight
            momentum_bias = eta_b * state.momentum_biases[index] - theta_b * grad_bias
            updated_weight = (1.0 - alpha_w) * weight + momentum_weight
            updated_bias = (1.0 - alpha_b) * state.biases[index] + momentum_bias
            new_weights.append(updated_weight)
            new_biases.append(updated_bias)
            new_momentum_weights.append(momentum_weight)
            new_momentum_biases.append(momentum_bias)

        return FastWeightState(
            weights=new_weights,
            biases=new_biases,
            momentum_weights=new_momentum_weights,
            momentum_biases=new_momentum_biases,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        state: FastWeightState | None = None,
    ) -> tuple[torch.Tensor, FastWeightState]:
        batch_size = hidden_states.size(0)
        if state is None:
            state = self.init_state(
                batch_size,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

        outputs = []
        for start in range(0, hidden_states.size(1), self.config.segment_len):
            end = min(start + self.config.segment_len, hidden_states.size(1))
            segment = hidden_states[:, start:end, :]
            normed_segment = self.pre_attn_norm(segment)
            memory_queries = self._pool_queries(normed_segment)
            retrieved_memory = self._memory_forward(memory_queries, state)
            attn_input = torch.cat([self._make_prefix(retrieved_memory), normed_segment], dim=1)
            attn_output = self._attention(attn_input)[:, self.config.prefix_tokens :, :]
            segment = segment + attn_output
            state = self._update_memory(segment, state)
            segment = segment + self.ffn(self.pre_ffn_norm(segment))
            outputs.append(segment)

        return torch.cat(outputs, dim=1), state


class TitansMACLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.blocks = nn.ModuleList([MACBlock(config) for _ in range(config.n_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        self._init_block_parameters()
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Conv1d):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _init_block_parameters(self) -> None:
        residual_std = 0.02 / math.sqrt(2 * self.config.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.persistent_memory, mean=0.0, std=0.02)
            nn.init.normal_(block.prefix_position, mean=0.0, std=0.02)
            nn.init.constant_(block.theta_head.bias, -3.0)
            nn.init.normal_(block.attn_out.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.ffn.down.weight, mean=0.0, std=residual_std)

    def init_memory_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> ModelState:
        return ModelState(
            layer_states=[
                block.init_state(batch_size, device=device, dtype=dtype) for block in self.blocks
            ],
            tokens_seen=0,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        labels: torch.Tensor | None = None,
        memory_state: ModelState | None = None,
        reset_memory: bool = False,
    ) -> TitansOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq_len].")

        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        hidden_dtype = self.token_embedding.weight.dtype

        if memory_state is None or reset_memory:
            memory_state = self.init_memory_state(
                batch_size,
                device=device,
                dtype=hidden_dtype,
            )
        elif len(memory_state.layer_states) != len(self.blocks):
            raise ValueError("Memory state does not match the number of model layers.")
        elif not self.training:
            memory_state = memory_state.enable_grad()

        position_ids = (
            torch.arange(seq_len, device=device) + memory_state.tokens_seen
        ) % self.config.max_seq_len
        hidden_states = self.token_embedding(input_ids) + self.position_embedding(position_ids)[None, :, :]

        next_states = []
        for block, block_state in zip(self.blocks, memory_state.layer_states):
            hidden_states, next_state = block(hidden_states, block_state)
            next_states.append(next_state)

        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shifted_logits = logits[:, :-1, :].contiguous()
            shifted_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shifted_logits.view(-1, shifted_logits.size(-1)),
                shifted_labels.view(-1),
            )

        next_model_state = ModelState(
            layer_states=next_states,
            tokens_seen=memory_state.tokens_seen + seq_len,
        )

        if not self.training:
            logits = logits.detach()
            next_model_state = next_model_state.detach()
            if loss is not None:
                loss = loss.detach()

        return TitansOutput(logits=logits, loss=loss, state=next_model_state)
