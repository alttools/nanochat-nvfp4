Below is a practical, “follow‑the‑breadcrumbs” set of edits to take **nanochat** from its current baseline to:

1. **NVFP4-style training path** (weight‑quantized, BF16 activations, STE backprop),
2. a **modernized Transformer block** (learnable RMSNorm + SwiGLU FFN + GQA + RoPE),
3. **TE fused attention** in place of vanilla SDPA (with clean fallbacks).

I’m writing this so a junior dev can do it safely in small, testable steps. I’ll point to exactly where the current code does things, then show the edits and how to wire them together. References to the repo you attached are called out inline. 

---

## 0) What’s already there (so you don’t re‑invent it)

Open `nanochat/gpt.py`:

* **RMSNorm** is already used, but **parameter‑free** via `F.rms_norm` wrapped in the helper `norm(x)` (no learnable γ). You’ll convert this to a learnable RMSNorm. 
* **RoPE** is implemented in `apply_rotary_emb()` and precomputed in the GPT module buffers; base θ is currently `10000` and the cache is `sequence_len * 10`. 
* **GQA/MQA** is already supported: the attention module has `n_head` and `n_kv_head` and duplicates K/V if needed. You’ll keep this and just expose a clearer config. 
* **FFN** uses **ReLU²** (`relu(x).square()`), in class `MLP`. You’ll swap it for **SwiGLU**. 
* **Attention** uses **PyTorch SDPA** (`F.scaled_dot_product_attention`) inside `CausalSelfAttention.forward`. You’ll route this through TE fused attention when possible and fall back to SDPA otherwise. 

Training, evaluation and web stack will continue to work once these are drop‑in replaced. The scripts call the model through `scripts/base_train.py`, `scripts/base_eval.py`, `scripts/chat_eval.py`, & friends. 

---

## 1) Dependency & environment setup

Edit `pyproject.toml` and add **Transformer Engine** (and keep it optional at import time in code):

```toml
[project]
dependencies = [
  # ... existing deps ...
  "transformer-engine>=1.6.0",  # adjust to whatever version your DGX image supports
]
```

> Why optional? Some devs won’t have TE locally; your code should gracefully fall back to vanilla SDPA. Your training scripts already handle BF16 autocast on CUDA. 

---

## 2) Add a small config surface (no big framework)

We’ll add a few toggles to the **model config** so scripts can opt in without changing a ton of code.

**File:** `nanochat/gpt.py`

Right under `@dataclass class GPTConfig`, add:

```python
@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768

    # new knobs
    attn_impl: str = "auto"  # "te" | "sdpa" | "auto"
    rope_base: int = 10000   # e.g., 10000 (current) or 1000000 for longer contexts
    norm: str = "rms"        # "rms" (learnable) ; keep "rms_fused" later if you add TE’s fused layernorm
    ffn: str = "swiglu"      # "relu2" (old) | "swiglu"
    precision: str = "bf16"  # "bf16" | "nvfp4"
    fp4_group_size: int = 128  # grouping for FP4 quant
```

You’ll thread these through the training scripts in step 7.

---

## 3) Implement **learnable RMSNorm**

Create a tiny module with a learnable `weight` (γ), and use the same RMS definition the model already likes.

**File:** `nanochat/rmsnorm.py` (new)

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # √(E[x^2]) over the last dim
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight
```

> This replaces the current parameter‑free `norm(x)` in `gpt.py`. Keep the function name `norm` for minimal code churn, but make it dispatch to `RMSNorm` instances you own.

**Edit `nanochat/gpt.py`:**

* Add `from nanochat.rmsnorm import RMSNorm` at the top.
* In `Block.__init__`, create **two** norm modules:

  ```python
  self.norm1 = RMSNorm(config.n_embd)
  self.norm2 = RMSNorm(config.n_embd)
  ```
* In `Block.forward`, replace calls to `norm(x)` with the new modules:

  ```python
  x = x + self.attn(self.norm1(x), cos_sin, kv_cache)
  x = x + self.mlp(self.norm2(x))
  ```

(This modernizes the block to a standard Pre‑Norm layout with learnable γ.)

---

## 4) Implement **SwiGLU** FFN

Replace ReLU² with SwiGLU. Typical hidden size is ~`(2/3)*4d = 2.666…d`. A simple, effective recipe:

**File:** `nanochat/gpt.py` — replace the `MLP` class:

```python
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.n_embd
        hidden = int(2.6667 * d)  # common choice for SwiGLU
        self.c_fc = nn.Linear(d, 2 * hidden, bias=False)  # gate + up
        self.c_proj = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        u, v = self.c_fc(x).chunk(2, dim=-1)
        return self.c_proj(torch.nn.functional.silu(u) * v)
```

This is drop‑in and fast. The rest of the model code need not change. (You can keep a guard so `ffn="relu2"` uses the original class if you like.)

---

## 5) NVFP4 “native” path (weight‑quantized training with STE)

There’s no first‑class `torch.float4` to train with today; the battle‑tested approach is **QAT‑style FP4 weights** + **BF16 activations** + **FP32 optimizer** with a **straight‑through estimator**. You quantize/dequantize **on the forward path only**; grads flow to the full‑precision master weights.

Create a small utility and a Linear wrapper:

**File:** `nanochat/quant4.py` (new)

```python
import torch
import torch.nn as nn

@torch.no_grad()
def quantize_fp4_symmetric_pergroup(w, group_size: int = 128):
    """
    Symmetric 4-bit per-group quantization.
    Returns (packed_uint8, scales) where packed_uint8 has 2 nibbles per byte.
    """
    orig_shape = w.shape
    w = w.contiguous().view(-1, orig_shape[-1])  # [rows, cols]
    cols = w.shape[1]
    assert cols % group_size == 0, "cols must be divisible by group_size"
    num_groups = cols // group_size

    w_reshaped = w.view(w.shape[0], num_groups, group_size)
    max_abs = w_reshaped.abs().amax(dim=-1, keepdim=True) + 1e-8
    scales = max_abs / 7.0  # 4-bit symmetric: levels -7..+7

    q = torch.clamp((w_reshaped / scales).round(), -7, 7).to(torch.int8)  # [-7,7]
    q = (q + 8).to(torch.uint8)  # shift to [1..15], reserve 0 if needed

    # pack 2 nibbles per byte
    hi = (q[..., ::2] & 0x0F)
    lo = (q[..., 1::2] & 0x0F)
    packed = (hi << 4) | lo  # [rows, groups, group_size//2]
    return packed.contiguous(), scales.squeeze(-1).contiguous()

def dequantize_fp4_symmetric_pergroup(packed, scales, group_size: int, cols: int):
    rows = packed.shape[0]
    num_groups = cols // group_size
    # unpack nibbles
    hi = (packed >> 4) & 0x0F
    lo = packed & 0x0F
    q = torch.empty(rows, num_groups, group_size, dtype=torch.int8, device=packed.device)
    q[..., ::2] = (hi.to(torch.int8) - 8)
    q[..., 1::2] = (lo.to(torch.int8) - 8)
    w = (q * scales.unsqueeze(-1)).view(rows, cols)
    return w

class FP4LinearSTE(nn.Module):
    """
    Drop-in nn.Linear replacement:
    - Keeps full-precision weight as Parameter (for the optimizer)
    - Forward path fake-quantizes to FP4 (on-the-fly) with STE
    - Bias stays in BF16/FP32
    """
    def __init__(self, in_features, out_features, bias=False, group_size=128, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        self.bias = nn.Parameter(torch.zeros(out_features, **factory_kwargs)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.group_size = group_size

    def forward(self, x):
        if not self.training:
            # eval: (optionally) cache packed weights; dequant on the fly
            w = self.weight
        else:
            w = self.weight

        # STE: quantize-dequantize on forward, let gradients flow to full-precision
        with torch.no_grad():
            packed, scales = quantize_fp4_symmetric_pergroup(w, self.group_size)
        w_q = dequantize_fp4_symmetric_pergroup(packed, scales, self.group_size, w.shape[1]).to(x.dtype)
        w_ste = w + (w_q - w).detach()  # straight-through
        return torch.nn.functional.linear(x, w_ste, self.bias)
```

**Wire it in**: In `nanochat/gpt.py`, add:

```python
from nanochat.quant4 import FP4LinearSTE
```

Change **every** `nn.Linear(...)` creation to:

```python
Linear = FP4LinearSTE if config.precision == "nvfp4" else nn.Linear

self.c_q   = Linear(self.n_embd, self.n_head   * self.head_dim, bias=False)
self.c_k   = Linear(self.n_embd, self.n_kv_head* self.head_dim, bias=False)
self.c_v   = Linear(self.n_embd, self.n_kv_head* self.head_dim, bias=False)
self.c_proj= Linear(self.n_embd, self.n_embd, bias=False)
# and in MLP:
self.c_fc  = Linear(config.n_embd, 2 * hidden, bias=False)
self.c_proj= Linear(hidden, config.n_embd, bias=False)
# and (optionally) the lm_head if you want it quantized too:
# self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)
```

**Why this design?**

* It keeps **optimizer states** and **master weights** exactly as before (FP32/AdamW, Muon), so you don’t break training stability.
* It simulates “native FP4 weight compute” during forward while reusing your entire training loop.
* Later, for deployment, you can pre‑pack and store the FP4 weights to avoid on‑the‑fly dequant.

---

## 6) TE fused attention (training path) with clean fallbacks

We’ll dispatch inside `CausalSelfAttention.forward`. Keep TE only for **training (no KV cache)**, and fall back to PyTorch SDPA when TE isn’t available or when you’re in **inference** mode (KV cache present).

**File:** `nanochat/gpt.py` (inside `CausalSelfAttention.forward`)

Add the import (top of file):

```python
try:
    import transformer_engine.pytorch as te
    _TE_AVAILABLE = True
except Exception:
    _TE_AVAILABLE = False
```

In `forward(...)`, **after** you compute `q,k,v` (already shaped to `[B, H, T, D]` and RoPE applied), add the branch:

```python
enable_gqa = self.n_head != self.n_kv_head
Tq, Tk = q.size(2), k.size(2)

use_te = (
    _TE_AVAILABLE
    and kv_cache is None      # simpler path: no cache, standard training fwd
    and (self.training)       # reserve for training
)

if use_te and (self.n_head == self.n_kv_head):
    # TE path for standard MHA; for GQA you can repeat k/v or keep SDPA fallback
    # TE expects [B, T, H, D]; we currently have [B, H, T, D], so transpose back
    q_ = q.transpose(1, 2).contiguous()
    k_ = k.transpose(1, 2).contiguous()
    v_ = v.transpose(1, 2).contiguous()

    # Most TE builds export a fused attention entry point; API signatures vary slightly by version.
    # Try the functional attention and fall back if not present.
    try:
        y_ = te.attention(q_, k_, v_, attn_bias=None, p=0.0, is_causal=True)  # TE fused path
        y = y_.transpose(1, 2).contiguous()
    except Exception:
        # TE not providing this symbol in your environment; use SDPA fallback
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

elif kv_cache is None or Tq == Tk:
    y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
elif Tq == 1:
    y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
else:
    # chunked decode path (already present in repo) – keep as-is
    attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device)
    prefix_len = Tk - Tq
    if prefix_len > 0:
        attn_mask[:, :prefix_len] = True
    attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
    y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)
```

> Notes
>
> * TE provides multiple attention kernels and the exact call can differ by version. The guarded `try/except` plus clean SDPA fallback keeps your tree green even when TE isn’t present or the symbol name differs.
> * If you want TE on **GQA** too, repeat K/V across heads (`repeat_interleave`) before the TE call; for now, SDPA fallback for GQA is perfectly fine and still fast on H100.

---

## 7) Wire flags into the training scripts

Expose **precision**, **attn_impl**, **rope_base**, **n_kv_head**, **ffn** in the scripts so you can switch configurations without editing code.

**File:** `scripts/base_train.py`

At the top where user settings live, add:

```python
# New toggles
attn_impl = "auto"     # "te"|"sdpa"|"auto"
precision = "bf16"     # "bf16"|"nvfp4"
rope_base = 10000
ffn = "swiglu"         # "swiglu"|"relu2"
gqa_kv_heads = -1      # -1 means keep default (== num_heads)
```

Allow CLI overrides via the existing configurator (already wired). 

Then, when you build `GPTConfig`, pass the new fields:

```python
num_kv_heads = num_heads if gqa_kv_heads == -1 else gqa_kv_heads
model_config_kwargs = dict(
    sequence_len=max_seq_len,
    vocab_size=vocab_size,
    n_layer=num_layers,
    n_head=num_heads,
    n_kv_head=num_kv_heads,
    n_embd=model_dim,
    attn_impl=attn_impl,
    rope_base=rope_base,
    norm="rms",
    ffn=ffn,
    precision=precision,
)
```

**Precision & autocast:** keep your existing BF16 autocast. With NVFP4 active, activations stay BF16; only weights are fake‑quantized in the linear wrappers, so no changes are required here. 

---

## 8) Bump RoPE base (optional, but recommended)

If you’re chasing longer contexts, bump the default `rope_base`:

**File:** `nanochat/gpt.py`, in `_precompute_rotary_embeddings`, swap:

```python
def _precompute_rotary_embeddings(self, seq_len, head_dim, base=None, device=None):
    base = self.config.rope_base if base is None else base
    # rest unchanged
```

And call with `base=self.config.rope_base`. Set `--rope_base=1000000` from the CLI when training deeper/longer models.

---

## 9) Checkpointing impact

You didn’t change the **state_dict** semantics for master weights (they’re still FP32/BF16 `nn.Parameter`s). So `save_checkpoint`/`load_checkpoint` keep working exactly the same. No special migration is needed. (If later you want true FP4 storage, add an export pass that packs weights and saves auxiliary `scales`.)

**Files:** `nanochat/checkpoint_manager.py` and scripts — no change needed. 

---

## 10) Quick tests (smoke first, then speed)

1. **Unit smoke**

   * Run a tiny config on CPU/MPS to confirm shapes:

     ```bash
     python -m scripts.base_train --depth=4 --max_seq_len=256 --device_batch_size=1 --num_iterations=2 --ffn=swiglu
     ```
   * Then the same with `--precision=nvfp4` to make sure `FP4LinearSTE` compiles.

2. **CUDA smoke**

   * BF16 baseline:

     ```bash
     torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=12 --ffn=swiglu --attn_impl=auto
     ```
   * NVFP4 path:

     ```bash
     torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=12 --precision=nvfp4 --ffn=swiglu --attn_impl=te
     ```

3. **Fused attention check**

   * Temporarily print which path fired (TE vs SDPA) inside `CausalSelfAttention.forward` with a rank‑0 guarded log.

4. **Bench à la repo**
   Your scripts already log MFU/tok‑sec and evals (`base_loss.py`, `base_eval.py`). Reuse those to compare speed and bpb after your changes. 

---

## 11) What each change optimizes and the trade‑offs

* **Learnable RMSNorm**
  *Optimizes:* Training stability and final loss (learnable γ helps fit).
  *Trade‑offs:* A few extra parameters and negligible compute.

* **SwiGLU FFN**
  *Optimizes:* Parameter‑efficiency and perplexity (SwiGLU beats ReLU² consistently for LLMs).
  *Trade‑offs:* Slightly more matmul size in the intermediate, roughly same wall‑time with fused kernels.

* **GQA (keep & expose)**
  *Optimizes:* KV memory and throughput at long sequences; better decode with fewer KV heads.
  *Trade‑offs:* Small quality hit if you over‑compress K/V; tune `n_kv_head`.

* **RoPE base bump**
  *Optimizes:* Context window stretching without architecture surgery.
  *Trade‑offs:* Small calibration changes; in practice fine.

* **NVFP4 fake‑quant weights (STE)**
  *Optimizes:* Memory footprint and matmul bandwidth; moves you toward FP4 deployment readiness.
  *Trade‑offs:* Slight extra overhead from quant/dequant in forward; accuracy depends on group size and scheme—start with per‑group symmetric (128) then experiment.

* **TE fused attention**
  *Optimizes:* Attention FLOPs via vendor‑tuned kernels on H100 (DGX).
  *Trade‑offs:* Adds a soft dependency; APIs differ across TE versions, so keep the SDPA fallback.

---

## 12) Minimal diffs (so you can paste confidently)

**`nanochat/gpt.py` — key replacements (illustrative):**

```python
# NEW imports
from nanochat.rmsnorm import RMSNorm
from nanochat.quant4 import FP4LinearSTE
try:
    import transformer_engine.pytorch as te
    _TE_AVAILABLE = True
except Exception:
    _TE_AVAILABLE = False

# pick linear
Linear = FP4LinearSTE if config.precision == "nvfp4" else nn.Linear

class CausalSelfAttention(nn.Module):
    def __init__(...):
        ...
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        ...
        # after q,k,v computed and rope+norm applied...
        use_te = _TE_AVAILABLE and self.training and (kv_cache is None) and (self.n_head == self.n_kv_head)
        if use_te:
            y_ = te.attention(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2),
                              attn_bias=None, p=0.0, is_causal=True)
            y = y_.transpose(1, 2)
        else:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=(self.n_head!=self.n_kv_head))

        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.n_embd
        hidden = int(2.6667 * d)
        self.c_fc = Linear(d, 2 * hidden, bias=False)
        self.c_proj = Linear(hidden, d, bias=False)
    def forward(self, x):
        u, v = self.c_fc(x).chunk(2, dim=-1)
        return self.c_proj(torch.nn.functional.silu(u) * v)

class Block(nn.Module):
    def __init__(...):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)
        self.norm1 = RMSNorm(config.n_embd)
        self.norm2 = RMSNorm(config.n_embd)
    def forward(self, x, cos_sin, kv_cache):
        x = x + self.attn(self.norm1(x), cos_sin, kv_cache)
        x = x + self.mlp(self.norm2(x))
        return x
```

**`scripts/base_train.py` — pass new config fields:**

```python
model_config_kwargs = dict(
    sequence_len=max_seq_len,
    vocab_size=vocab_size,
    n_layer=num_layers, n_head=num_heads,
    n_kv_head=num_kv_heads, n_embd=model_dim,
    attn_impl=attn_impl, rope_base=rope_base,
    norm="rms", ffn=ffn, precision=precision,
)
```

That’s the core of it.

---

## 13) Sanity checklist (you can copy/paste into your PR)

* [ ] `rmsnorm.py` added; `Block` uses `norm1/norm2`.
* [ ] `MLP` is SwiGLU; flag allows returning to ReLU² if needed.
* [ ] `quant4.py` added; `FP4LinearSTE` wraps all Linear layers when `precision="nvfp4"`.
* [ ] `CausalSelfAttention` tries TE fused attention for **training** when available; otherwise SDPA.
* [ ] Config fields surfaced in `GPTConfig` and plugged in by `base_train.py`.
* [ ] BF16 autocast unchanged; optimizer states unchanged; checkpoints load/store identical.
* [ ] Speed & loss verified on a tiny run; then an 8xH100 run.

---

Curiosity never stops at “it runs.” Once you’ve got this landed, benchmark **group size** (64/128/256), **GQA ratios** (e.g., `n_head=16, n_kv_head=4`), and **rope_base** (10k vs 1M) on a fixed token budget. The sweet spot on DGX‑class H100 boxes is often surprising, and now you’ve got the switches to find it. 
