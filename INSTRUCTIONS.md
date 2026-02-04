# Mission

Implement an optional attention backend for diffusion transformer inference in
ComfyUI that replaces Softmax attention’s exponential kernel with a **truncated
Taylor expansion** and evaluates it via a **symmetry-aware polynomial feature
map** (upper-hyper-triangular monomial basis), as described in the paper
*“Self-Attention at Constant Cost per Token via Symmetry-Aware Taylor
Approximation”*. 

Primary target: **self-attention over image/latent tokens** at high resolution
(large token counts). Secondary target: **cross-attention** (text
conditioning), if convenient.

Note: You can find ComfyUI in ../ComfyUI if you wish to browse the source code.

Constraints:

* Must be implemented as a ComfyUI custom node using the v3 node API.
* Must be **switchable** at runtime (flag / config).
* Must keep a **fallback path** to standard attention.
* Must include correctness tests and a resolution-dependent benchmark.

---

# 0) Read-these-first: the essential math you’re implementing

You don’t need to re-derive the paper; just implement these pieces:

1. Kernel approximation (paper Eq. 3 + truncation Eq. 9):
   [
   \exp(\text{score}) \approx \sum_{p=0}^{P-1}\frac{\text{score}^p}{p!}
   ]
   In attention, (\text{score} = (q^\top k)\cdot \text{scale}) where typically (\text{scale}=1/\sqrt{d}). The paper writes (\exp(q^\top k / c)) with (c=\sqrt{d}). Same thing. 

2. Symmetry-aware monomial basis (paper Eq. 7, 10, 11):
   For each degree (p), build the feature vector (\Phi_p(x)) containing the **unique** degree‑(p) monomials corresponding to combinations-with-replacement of indices (i_1\le \dots \le i_p).
   The number of features is (m_p=\binom{d+p-1}{p}). 
   They compute it with a precomputed index matrix (M_p) and `prod` over gathered elements. 

3. Multiplicity weighting (C_p) (paper Eq. 8):
   Because you only keep unique monomials, each one has a multiplicity coefficient equal to the number of permutations that produce it. This is the diagonal of (C_p). 

4. Linear-attention style evaluation (paper Appendix A2):
   For causal attention they do a recurrence; for diffusion self-attn (non-causal) you can do global sums instead. The form you want is:

* Build accumulated key/value statistics in feature space
* Then evaluate all queries against those statistics
  This avoids ever forming an (N\times N) attention matrix. 

---

# 1) Repo reconnaissance: locate Flux.2 attention in ComfyUI

**Goal:** find the exact function/class that computes attention for Flux.2 inference.

Do not guess file paths; use search.

### 1.1 Find attention entrypoints (ripgrep targets)

Run ripgrep for any/all of these:

* `scaled_dot_product_attention`
* `xformers`
* `memory_efficient_attention`
* `flash_attn`
* `attention`
* `attn`
* `qkv`
* `to_qkv`
* `rope` / `rotary` / `pos_bias`
* `Flux` / `flux`

Identify:

* The module(s) implementing MHA for Flux.2 and other transformer-based models in ComfyUI.
* Whether attention is implemented via:

  * PyTorch SDPA (`torch.nn.functional.scaled_dot_product_attention`)
  * xFormers
  * custom kernel
  * explicit `softmax(qk)` code

### 1.2 Record the exact tensor shapes

In the attention forward path, log (or inspect) shapes and dtypes:

* `q`, `k`, `v` shapes (common are `[B, H, N, d]` or `[B, N, H, d]`)
* `mask` shape if present (padding mask, etc.)
* head dimension `d`
* `scale` used (is it `1/sqrt(d)`? or something else?)
* any q/k normalization (RMSNorm, L2 norm, “qk_norm”, etc.)
* any additive attention bias terms (`attn_bias`, relative position bias)

**Critical:** if the implementation adds a per-pair bias (b_{ij}) inside the exponent (e.g. (\exp(q_i^\top k_j + b_{ij}))), your feature-factorization may not apply cleanly. You can still proceed if Flux.2 uses only dot-product + scale (or something decomposable). If not sure, add a “not supported” check and fallback.

### 1.3 Identify where to insert a backend switch

Look for:

* a centralized `attention()` helper
* an `Attention` / `MultiHeadAttention` class
* a backend selector e.g., use xFormers if available)

You want a single hook point like:

```python
if backend == "taylor":
    return taylor_attention(q,k,v,mask, ...)
else:
    return standard_attention(q,k,v,mask, ...)
```

---

# 2) Design: what you will implement

You are implementing an **approximate Softmax attention** by approximating the exponential kernel, not changing the overall attention formula. 

For each query position (i):
[
y_i \approx \frac{\sum_j \hat{e}(q_i,k_j) v_j}{\sum_j \hat{e}(q_i,k_j)}
]
where (\hat{e}(q,k)) is the Taylor approximation to (\exp((q^\top k)\cdot scale)).

### Key implementation decision (recommended)

Implement **one concatenated feature map** so the whole kernel becomes a single dot product:

Define for each degree (p):

* multiplicity weight (w_{p,r}) (diag of (C_p))
* Taylor coefficient: (\beta_p = \frac{scale^p}{p!}) (equivalent to paper’s (\alpha_p)) 
* feature vector:
  [
  \psi_p(x) = \sqrt{\beta_p},\sqrt{w_p},\Phi_p(x)
  ]

Then:
[
\hat{e}(q,k) = \sum_p \beta_p \langle \Phi_p(q),\Phi_p(k)\rangle_{C_p}
= \langle \psi(q), \psi(k)\rangle
]
with (\psi(x)=\text{concat}_p \psi_p(x)).

This trick collapses “multiple Taylor terms” into a single linear-attention feature space without changing the math.

---

# 3) Implement the symmetry-aware feature machinery

Create a small module, independent of ComfyUI specifics, e.g. `taylor_sym_features.py`.

## 3.1 Precompute index matrices (M_p)

For each degree (p\in[0,P-1]), build `M_p` of shape `(m_p, p)` where each row is a nondecreasing tuple of indices in `[0, d-1]`.

* Use combinations-with-replacement.
* Cache by `(device, d, P)` or `(d, P)` and move to device as needed.

Special case (p=0): `M_0` is an empty index list; (\Phi_0(x)=1). 

## 3.2 Precompute multiplicity weights (w_p) (diag of (C_p))

For each row (a multiset of indices), the multiplicity is:
[
w = \frac{p!}{\prod_{u} c_u!}
]
where (c_u) are counts of repeated indices in that row.

Store:

* `w_p` (float tensor shape `[m_p]`)
* you will use `sqrt_w_p = sqrt(w_p)`.

## 3.3 Feature evaluation (paper Eq. 10–11)

Implement:

```python
# x: (..., d)
# M_p: (m_p, p) long
# returns (..., m_p)
Phi_p(x) = x[..., M_p].prod(dim=-1)
```

Then apply scaling:

```python
psi_p(x) = Phi_p(x) * sqrt_w_p * sqrt_beta_p
```

Notes:

* The paper highlights PyTorch advanced indexing often copies data (bandwidth hit). That’s okay for a first proof-of-concept; optimize later if it works. 
* Keep the feature computation in fp32 (or at least accumulate in fp32), because products can under/overflow in fp16/bf16.

---

# 4) Implement Taylor linear attention (non-causal, diffusion-friendly)

Diffusion self-attention is typically **non-causal** over tokens. The paper’s Appendix A2 gives the causal scan form; for non-causal you do “global sums” instead of prefix sums. 

## 4.1 Compute two sufficient statistics from keys/values

Let:

* `phi_k = ψ(K)` shape `[B, H, Nk, R]` where `R = sum_p m_p`
* `V` shape `[B, H, Nk, dV]`

Compute:

* `S = Σ_j phi_k[j]^T * v_j` → shape `[B, H, R, dV]`
* `Z = Σ_j phi_k[j]` → shape `[B, H, R]`

If there’s a padding mask `mask[j]` in `{0,1}`:

* incorporate by multiplying contributions by `mask[j]`.

## 4.2 Evaluate outputs for queries

For each query token:

* `phi_q = ψ(Q)` shape `[B,H,Nq,R]`
* numerator: `num = phi_q @ S` → `[B,H,Nq,dV]`
* denominator: `den = phi_q @ Z` → `[B,H,Nq]`
* output: `out = num / (den[...,None] + eps)`

## 4.3 Do it in blocks to control memory

Do **not** materialize `phi_k` or `phi_q` for all tokens if `R` is large.

Recommended approach:

* Pass 1: iterate over `K,V` in blocks of tokens, accumulate `S` and `Z`.
* Pass 2: iterate over `Q` in blocks, compute outputs using `S,Z`.

This keeps peak memory ~`O(B*H*R*dV)` plus a block.

## 4.4 Cross-attention support (nice win)

Cross-attention is even easier:

* keys/values are from the conditioning sequence (text tokens)
* queries are image tokens
  Compute `S,Z` once from K/V, then evaluate all Q blocks.

---

# 5) Causal variant (optional)

If you discover Flux.2 uses causal attention anywhere (unlikely for image diffusion, but check), implement the scan recurrence from Appendix A2:

* maintain per-step `Z_t` and `S_t`
* update with each new token
* evaluate query with `ψ(q_t)` each step 

This is optional unless the codebase actually needs causal mode.

---

# 6) Integration into ComfyUI / Flux.2

## 6.1 Add a backend flag

Implement at least one of:

* environment variable: `COMFY_ATTENTION_BACKEND=taylor`
* config option in model loader
* a ComfyUI “node” option (if that’s how Flux.2 backend selection is done)

Also add parameters:

* `P` (default 4, per paper’s claim that 4 terms often hit ~float16-scale errors) 
* `switch_threshold_tokens` (e.g. only use Taylor when `N >= 10000`, else standard)
* `max_feature_dim_R` or `max_head_dim` safety guard

## 6.2 Fallback rules (important for stability)

The truncated Taylor polynomial can produce:

* negative kernel contributions (in principle),
* very small/negative denominators.

So implement:

* `den_clamped = clamp(den, min=eps)` OR
* if `any(den <= eps)` or `any(isnan(out))`: fallback to standard attention for that layer call.

Log counters so you can see if fallback happens often.

## 6.3 Preserve exact scaling behavior

Use the exact same `scale` that the existing attention path uses. Don’t assume `1/sqrt(d)`; read it from code and reuse.

---

# 7) Correctness tests (must-have)

## 7.1 Unit tests vs standard attention

Write a test file that:

* samples random `q,k,v` with matching shapes and dtype
* compares:

  * baseline attention output (whatever Flux.2 path uses)
  * Taylor attention output (your implementation)
* run across:

  * multiple token lengths: 64, 256, 1024, 4096 (and higher if feasible)
  * multiple head dims: the model’s real head dim, plus smaller ones (16/32) for sanity
  * with/without masks

Report:

* max abs error
* mean abs error
* relative error where denom is large

Don’t expect bitwise match. You need “small enough that generation doesn’t collapse.” Start with a target like “within a few e‑2 in fp16 space” and tighten if feasible. The paper’s synthetic tests suggest ~float16-scale elementwise errors at `P=4`. 

## 7.2 End-to-end sanity in ComfyUI

Pick a small, fast pipeline:

* fixed seed
* small resolution (where standard attention works)
  Run with:
* standard backend
* Taylor backend (even if it’s slower here)

Compare:

* latent statistics (mean/std)
* final image perceptual similarity (basic SSIM/LPIPS if you have it; otherwise just ensure “not broken”)

---

# 8) Performance benchmarks (must-have)

## 8.1 Microbench attention layer

Create a benchmark script that times just the attention call:

* sweep token count `N` by simulating various resolutions / patch counts
* measure:

  * wall time
  * peak CUDA memory
    Compare:
* standard attention (current backend)
* Taylor backend

Expect crossover only when `N` is “large”; paper shows big wins at long context but acknowledges their PyTorch implementation has overheads and needs kernel work for best results. 

## 8.2 Full denoise step benchmark

Run one denoise step (one model forward) at increasing resolutions:

* log total time and memory
* verify attention is indeed the bottleneck (use `torch.profiler`)

---

# 9) Known gotchas you must explicitly check

1. **Is there an additive attention bias?**
   If attention logits include `+ bias[i,j]` inside the exponent, the clean “feature dot product” factorization may not apply. If present, either:

* fallback to standard attention, or
* implement only where bias is absent.

2. **Head dim matters a lot**
   The feature dimension (R=\binom{d+P-1}{P-1}) grows quickly with `d`. The method is most favorable for small heads (paper discussion + cost formulas). 
   So implement guards and measure `R` at runtime.

3. **Existing backends are already highly optimized**
   ComfyUI may be using FlashAttention/xFormers. Your Python-level gather+prod will not beat those at small/medium N. The initial win you’re hunting is “very large N”.

4. **Two-pass requirement for non-causal self-attn**
   For self-attn, you can’t produce outputs until you’ve accumulated `S,Z` over all keys. That’s fine, but it means your kernel is structured differently than causal scan.

---

# 10) Deliverables checklist

* [ ] `taylor_sym_features.py` (precompute `M_p`, `w_p`, feature eval)
* [ ] `taylor_attention.py` (non-causal linear attention using concatenated features; supports mask)
* [ ] Integration patch: backend switch in Flux.2 attention callsite
* [ ] Unit tests comparing to baseline attention
* [ ] Benchmark script(s) + logged results at multiple token counts
* [ ] Safety fallback path + logging counters
* [ ] Documentation: how to enable, set `P`, set threshold

