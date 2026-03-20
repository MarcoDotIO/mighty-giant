Yes. The pairing is sensible, but the clean merge is **hierarchical**, not equation-level. I would use **Mamba-3 as the fast token-level backbone** and **Titans as a sparse external long-term memory system**. That exploits the strongest result from each paper: Mamba-3 improves the quality/latency frontier of recurrent sequence modeling, while Titans fixes the fixed-state long-context recall problem that recurrent backbones still have. Reviewed papers: *Mamba-3: Improved Sequence Modeling using State Space Principles* and *Titans: Learning to Memorize at Test Time*.  

Mamba-3’s contribution is mostly “inside the recurrent block”: exponential-trapezoidal discretization adds a more expressive recurrence and can replace the short causal convolution; the complex-valued / RoPE-equivalent transition fixes state-tracking failures that earlier linear models had; and MIMO improves arithmetic intensity so quality goes up without a comparable decode-latency penalty. At 1.5B, the paper reports the MIMO variant improving average downstream accuracy over the SISO version, while decode latency stays close to other recurrent baselines; it also shows that pure Mamba-3 still remains weaker than attention-heavy models on some retrieval/extraction settings, and its own hybrid experiments improve retrieval. 

Titans solves the orthogonal problem: it adds a **test-time-updated neural long-term memory** with momentum and forget-gated updates, plus persistent memory. Its results suggest that **MAC** is best for hard long-context recall, **MAG** is usually the best cheaper fusion pattern for standard LM/reasoning, and deeper memory helps but costs throughput roughly linearly. The ablations also indicate that forgetting/weight decay and momentum matter more than convolution. 

## Recommended combined architecture

1. **Backbone: mostly Mamba-3, not Titans LMM**

   Make Mamba-3 the main per-token processor. Use the full Mamba-3 block: exponential-trapezoidal update, complex/data-dependent rotary transition, BCNorm, and B/C bias. Do **not** reintroduce the short conv in the backbone. Use **SISO in most layers**, and place **MIMO only in the upper quarter of layers** or every 4th block. That keeps most of Mamba-3’s decode advantage while still harvesting some of the MIMO quality gain. This follows the Mamba-3 evidence that SISO is the latency floor and MIMO is a targeted quality boost. 

2. **External long-term memory: Titans-style, but chunked and compact**

   Add **one global long-term memory module per sequence**, updated every segment, not every token. The key change I would make relative to Titans is this: **do not mutate a full-width memory MLP per request**. Instead, keep a shared static memory trunk and update only a **small low-rank fast-weight adapter** per active sequence. That preserves the Titans idea of learning to memorize at test time, but keeps serving memory practical.

   A good default is:

   * segment length: **256 or 512 tokens**
   * memory depth: **2 layers**
   * fast-weight rank: **16 or 32**
   * one shared memory module for the whole stack, or one per stage if the model is large

   This choice is directly motivated by Titans’ finding that deeper memory helps but slows linearly. 

3. **Memory read path: cheap MAG everywhere, sparse MAC occasionally**

   This is the important compromise.

   For each segment (S_i), retrieve a small bank of memory vectors (r_i) from the Titans memory using a query derived from the segment prefix or previous segment summary. Then fuse it in two ways:

   * **Cheap path (every block):** MAG-style gated residual injection
     (h \leftarrow \text{Mamba3}(h) + \sigma(W_g[h;\bar r_i]) \odot W_r \bar r_i)
   * **Expensive path (sparse):** every 6–8 blocks, insert a **small fusion block** that runs local attention or cross-attention over:

     * current hidden states
     * **4–8 retrieved memory slots**
     * **8–16 persistent memory slots**

   This borrows MAC’s stronger long-context behavior without paying MAC’s cost everywhere. Since Mamba-3’s hybrid results show attention helps retrieval, and Titans shows MAC is strongest on long-context tasks while MAG is the cheaper path, this hybrid-of-hybrids is the best balance. I would also keep the **pre-gate grouped RMSNorm** on these fusion blocks, because Mamba-3 found that norm placement matters in hybrids for long-context extrapolation.  

4. **Memory write path: write summarized, high-surprise states**

   Do not write raw token embeddings into the Titans memory. Let the Mamba-3 backbone process the segment first, then compute a small set of write candidates from the **post-backbone hidden states**. Use Titans’ surprise signal to select only the most informative states in the segment, then update the fast weights once per segment:

   [
   S_i = \eta_i S_{i-1} - \theta_i \nabla | M_{i-1}(k_i) - v_i |^2,\qquad
   M_i = (1-\alpha_i)M_{i-1} + S_i
   ]

   Use **segment-level** (\alpha_i,\eta_i,\theta_i), not token-level, for the first implementation. Titans explicitly notes chunk-level simplifications as a speed lever, and it is the right choice here because Mamba-3 is already handling fine-grained token dynamics. 

## Why this combination is the right compromise

The backbone remains **streaming and recurrent**, so decode stays close to Mamba-3. The Titans component is **amortized over segments**, so its cost does not hit every token. Long-range recall no longer depends on cramming everything into Mamba’s fixed recurrent state, which is exactly where pure Mamba-style models still struggle. And because the external memory is updated sparsely and stored as low-rank fast weights, you avoid the main deployment problem of a naïve Titans implementation: too much mutable per-request state.

I would also drop Titans’ per-branch convolution in the first combined version. Mamba-3 already gets convolution-like behavior from exponential-trapezoidal discretization plus B/C bias, while Titans’ own ablation suggests the bigger wins come from forgetting and momentum.  

## First prototype I would actually build

For a serious first pass, I would train this:

* **400M–800M scale**
* **75% Mamba-3 SISO blocks**
* **25% Mamba-3 MIMO blocks near the top**
* **d_state = 64 or 96**
* **segment length = 256**
* **memory depth = 2**
* **fast-weight rank = 16**
* **persistent slots = 16**
* **retrieved memory slots = 4**
* **one sparse fusion block every 8 layers**

Training should be staged:

1. pretrain the pure Mamba-3 backbone on 2K–4K context,
2. attach the Titans memory and persistent slots,
3. continue training with a context curriculum to 16K, 64K, then longer,
4. measure both **prefill/decode latency** and **NIAH/BABILong-style recall**.

The minimum ablations worth running are: MAG-only vs sparse MAG+MAC, memory depth 1/2/3, update frequency 128/256/512, and MIMO placement none/top-quarter/all.

The main risk is engineering complexity in serving: the Titans memory must be treated as **sequence state**, not as globally mutated model weights. If that is handled correctly, this combined model has a strong chance of being better than either paper’s architecture alone: Mamba-3 gives you the fast recurrent core, and Titans gives you the missing long-horizon memory.

