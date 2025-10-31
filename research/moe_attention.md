Below is an implementation plan to add **Mixture‑of‑Experts (MoE) on the Attention layer** to nanochat. It is written so a junior developer can follow it and land a working, testable feature in small, reviewable steps. The plan assumes PyTorch 2.8+ (as in nanochat) and the existing model/training codepaths. Where helpful, exact files/functions are cited.

---

## 0) What you’re grafting MoE into (quick inventory)

* **Attention today**: `CausalSelfAttention` in `nanochat/gpt.py` projects Q, K, V; applies RoPE; supports both training and KV‑cached inference; calls `F.scaled_dot_product_attention` with either `is_causal=True` or an explicit `[Tq × Tk]` mask for the “prefix + causal chunk” case; then `c_proj` back to residual stream. See the three attention branches and the mask construction. 
* **Block today**: `Block.__init__` wires `self.attn = CausalSelfAttention(...)` and calls it inside the residual path. This is the single call site to replace when MoE is enabled. 
* **KV cache shape**: `KVCache.insert_kv(layer_idx, k, v)` stores/returns views with shape `(B, H, T, D)` per layer. We’ll keep this API and only change where/how we **read** from it for expert‑specific sparsity windows. 
* **SFT training loop**: `scripts/chat_sft.py` calls `loss = model(train_inputs, train_targets)` in a standard loop; you can add auxiliary losses transparently by having the model return a scalar loss that already includes MoE regularizers/distillation. 
* **Project deps**: torch 2.8.0+; optional Triton can be added later for a fused sparse kernel. 

---

## 1) High‑level design

### 1.1 Goals

* Add **Adaptive Attention MoE** that performs attention over **expert‑specific sparse key sets** and mixes expert outputs per token:

  [
  y_t ;=; \sum_{i=1}^{E} \alpha_{t,i};\text{SDPA}*i!\big(q_t,, K*{(k_i)},, V_{(k_i)}\big)
  ]

  where each expert uses the same Q/K/V weights but a **different sparsity level** (k_i). The router produces per‑token expert weights (\alpha_{t,i}) (hard or soft).

* Two stages of training:

  1. **Distillation** at attention output (MSE to dense attention outputs; base weights frozen).
  2. **Fine‑tuning** (unfreeze weights; add load‑balancing regularizer).

* Keep **KV cache** and **GQA/MQA** behavior intact; keep Block/Engine interfaces unchanged for callers.

### 1.2 Tradeoffs for v1

* **Top‑k selection**: for v1, implement **local causal windows** per expert (e.g. (k={32,64,128})) instead of a global top‑k over all keys. This gives the O(N·k) compute/memory behavior **without** writing a custom kernel immediately. (The summary allows locality window as a valid selection rule.)
* **Soft vs. hard routing**: support both; default **hard routing with straight‑through** estimator (Gumbel-Softmax optional).
* **No parameter‑per‑expert duplication**: all experts share the block’s Q/K/V linear layers; experts differ **only** in which keys they view. This adds only the router parameters.

---

## 2) Configuration: extend `GPTConfig`

Add these fields to `nanochat/gpt.py` `GPTConfig`:

```python
@dataclass
class GPTConfig:
    # existing...
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    # --- new MoE on attention ---
    moe_attn: bool = False                 # turn feature on/off
    moe_num_experts: int = 3               # usually 3 (Peripheral/Focal/Reflective)
    moe_k: tuple[int, ...] = (32, 64, 128) # local causal windows per expert
    moe_router_hidden: int = 128           # router MLP width
    moe_router_temp_init: float = 2.0      # initial routing temperature
    moe_router_temp_min: float = 0.3       # floor temp (inference)
    moe_router_gumbel: bool = True         # use Gumbel-Softmax noise
    moe_router_hard: bool = True           # hard (one-expert) routing with ST
    moe_load_balance_lambda: float = 0.01  # expert usage regularizer
    moe_distill_lambda: float = 0.5        # weight for attention-output MSE in phase 1
    moe_window_selection: str = "local"    # "local" (v1); can add "topk" later
    moe_topk_per_head: bool = False        # future: per-head selection
```

---

## 3) Routing module

Create `nanochat/moe_attention.py` and implement a small router:

```python
# nanochat/moe_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Router(nn.Module):
    def __init__(self, d_model: int, num_experts: int, hidden: int):
        super().__init__()
        self.num_experts = num_experts
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden, bias=False),
            nn.SiLU(),
            nn.Linear(hidden, num_experts, bias=False),
        )
        self.register_buffer("temperature", torch.tensor(1.0), persistent=False)

    def set_temperature(self, t: float):
        self.temperature.fill_(max(1e-4, t))

    def forward(self, x, hard=True, use_gumbel=True):
        # x: (B, T, d_model), already pre-normed by Block
        logits = self.net(x) / self.temperature
        if use_gumbel:
            g = -torch.log(-torch.log(torch.rand_like(logits).clamp_(min=1e-10)))
            logits = logits + g
        probs = torch.softmax(logits, dim=-1)  # (B, T, E)
        if not hard:
            return probs, None  # soft routing
        # Straight-through hard routing
        idx = torch.argmax(probs, dim=-1)                     # (B, T)
        onehot = F.one_hot(idx, num_classes=probs.size(-1))   # (B, T, E)
        hard_st = onehot + (probs - probs.detach())           # ST trick
        return hard_st, idx
```

---

## 4) Adaptive MoE Attention module

### 4.1 Wire it into the Block

Replace the attention module in `Block` **only when enabled**:

```python
# nanochat/gpt.py
class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        if config.moe_attn:
            from nanochat.moe_attention import AdaptiveAttentionMoE
            self.attn = AdaptiveAttentionMoE(config, layer_idx)
        else:
            self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache):
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        x = x + self.mlp(norm(x))
        return x
```

> `Block` currently calls `self.attn(norm(x), cos_sin, kv_cache)`; your MoE module must keep the same `(x, cos_sin, kv_cache) -> y` signature. 

### 4.2 Implement `AdaptiveAttentionMoE`

Key points:

* Reuse the same Q/K/V projections as dense attention.
* Compute router outputs on **the pre‑normed `x`** provided by the Block.
* For **hard routing**, group tokens by expert and run SDPA once per expert over **that expert’s local window**.
* For **soft routing**, compute all experts’ outputs for all tokens, then weighted sum.

```python
# nanochat/moe_attention.py (continued)
class AdaptiveAttentionMoE(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0

        # shared QKV and output
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

        # router
        self.router = Router(self.n_embd, config.moe_num_experts, config.moe_router_hidden)
        self.moe_num_experts = config.moe_num_experts
        self.k_list = list(config.moe_k)
        self.hard = config.moe_router_hard
        self.use_gumbel = config.moe_router_gumbel
        self.window_selection = config.moe_window_selection
        self.load_balance_lambda = config.moe_load_balance_lambda
        self.distill_lambda = config.moe_distill_lambda

        # stats from last forward (for logging/loss)
        self.last_usage = None
        self.last_router_entropy = None
        self.last_distill_mse = None

    @torch.no_grad()
    def set_router_temperature(self, t: float):
        self.router.set_temperature(t)

    def _local_causal_mask(self, Tq, Tk, k, device):
        # Builds a boolean [Tq, Tk] mask that keeps only the last k tokens allowed by causality.
        # supports both training (Tk==Tq) and "prefix + chunk" (Tk >= Tq) cases seen in dense attention. :contentReference[oaicite:6]{index=6}
        mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=device)
        prefix_len = Tk - Tq  # can be 0 in training
        # q position j can see keys [:prefix_len + j] (causal)
        # restrict to last k of those
        for j in range(Tq):
            end = prefix_len + j + 1
            start = max(0, end - k)
            mask[j, start:end] = True
        return mask

    def _sdpa(self, q, k, v, attn_mask=None, enable_gqa=False):
        # q: (B, H, Tq, D), k/v: (B, Hkv, Tk, D)
        if attn_mask is None:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
        else:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()
        # 1) QKV
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # 2) RoPE + QK norm (mirror dense path) :contentReference[oaicite:7]{index=7}
        cos, sin = cos_sin
        from nanochat.gpt import apply_rotary_emb, norm
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # (B, H, T, D) etc.

        # 3) KV cache integration (mirror dense path): insert, get full views so far :contentReference[oaicite:8]{index=8}
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq, Tk = q.size(2), k.size(2)
        enable_gqa = self.n_head != self.n_kv_head

        # 4) Router (on token stream)
        router_probs, hard_idx = self.router(x, hard=self.hard, use_gumbel=self.use_gumbel)
        # Stats for regularization/logging
        with torch.no_grad():
            self.last_usage = router_probs.mean(dim=(0,1))       # (E,)
            p = router_probs.clamp_min(1e-9)
            self.last_router_entropy = (-p * p.log()).sum(dim=-1).mean().detach()

        # 5) Dense teacher (for distillation): reuse dense attention path but don't create masks
        y_dense = None
        if self.training and self.distill_lambda > 0:
            if kv_cache is None or Tq == Tk:
                y_dense = self._sdpa(q, k, v, attn_mask=None, enable_gqa=enable_gqa)
            elif Tq == 1:
                y_dense = self._sdpa(q, k, v, attn_mask=None, enable_gqa=enable_gqa)
            else:
                # prefix + causal chunk as in dense attention :contentReference[oaicite:9]{index=9}
                attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device)
                prefix_len = Tk - Tq
                if prefix_len > 0:
                    attn_mask[:, :prefix_len] = True
                attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
                y_dense = self._sdpa(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)

        # 6) Expert computations
        # We implement both paths; v1 default is hard routing for efficiency.
        if self.hard:
            # scatter-gather by expert
            y = torch.zeros_like(q)
            for e, k_e in enumerate(self.k_list):
                # select tokens for expert e
                idx_e = (hard_idx == e)  # (B, T)
                if not torch.any(idx_e):
                    continue
                # Build attention mask for expert window (per-batch, shared across heads)
                attn_mask = self._local_causal_mask(Tq, Tk, k_e, device=q.device)
                # compute full SDPA but we’ll zero out outputs for non-selected tokens
                y_e = self._sdpa(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)
                # y_e: (B, H, Tq, D). Keep only rows (tokens) routed to e:
                # broadcast idx_e (B,T) -> (B,1,T,1)
                sel = idx_e[:, None, :, None]
                y = torch.where(sel, y_e, y)
        else:
            # soft: compute y for each expert, then weight sum
            y_mix = 0.0
            for e, k_e in enumerate(self.k_list):
                attn_mask = self._local_causal_mask(Tq, Tk, k_e, device=q.device)
                y_e = self._sdpa(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)  # (B,H,T,D)
                alpha_e = router_probs[..., e]  # (B, T)
                y_mix = y_mix + y_e * alpha_e[:, None, :, None]
            y = y_mix

        # 7) Distillation loss at attention output
        self.last_distill_mse = None
        if self.training and self.distill_lambda > 0 and y_dense is not None:
            self.last_distill_mse = (y - y_dense).pow(2).mean()

        # 8) join heads and project
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y
```

**Why the “boolean mask SDPA” first?**
It is the minimal change that matches the existing dense codepaths and preserves correctness (including the prefix+chunk inference path). It can be replaced by a fused kernel later for memory/time wins.

> The dense path’s three branches (training; single‑token decode; prefix+chunk decode with explicit mask) are mirrored here to keep functionality parity with `CausalSelfAttention`. 

---

## 5) Model‑level integration (losses, temperature)

### 5.1 Sum auxiliary losses inside `GPT.forward`

Keep the external API stable: the training script expects `model(inputs, targets)` to return a **single scalar loss**. Add MoE penalties inside.

Implementation sketch in `nanochat/gpt.py`:

* Thread an accumulation for:

  * **Load balancing**: encourage average per‑token expert usage to be ~uniform.
  * **Distillation**: sum of `attn.last_distill_mse` across blocks (when enabled).

```python
# inside GPT.forward, after computing logits and CE loss
moe_lb = 0.0
moe_distill = 0.0
if getattr(self.config, "moe_attn", False) and self.training:
    for block in self.transformer.h:
        attn = block.attn
        if hasattr(attn, "last_usage") and attn.last_usage is not None:
            usage = attn.last_usage  # (E,)
            E = usage.numel()
            moe_lb = moe_lb + (usage - (1.0 / E)).pow(2).sum()
        if hasattr(attn, "last_distill_mse") and attn.last_distill_mse is not None:
            moe_distill = moe_distill + attn.last_distill_mse

loss = CE_loss
if self.training and getattr(self.config, "moe_attn", False):
    loss = loss + self.config.moe_load_balance_lambda * moe_lb \
                + self.config.moe_distill_lambda * moe_distill
return loss
```

> This preserves the current training loop in `scripts/chat_sft.py` which expects a scalar `loss` back. 

### 5.2 Temperature scheduling hooks

* Add a method on `GPT` to set router temperature across all MoE blocks:

```python
def set_moe_router_temperature(self, t: float):
    if not getattr(self.config, "moe_attn", False):
        return
    for block in self.transformer.h:
        if hasattr(block.attn, "set_router_temperature"):
            block.attn.set_router_temperature(t)
```

* In **distillation phase**, start higher (`moe_router_temp_init`) and **anneal** toward `moe_router_temp_min`. In **inference**, set directly to `moe_router_temp_min`.

---

## 6) Loading a base checkpoint **with** MoE enabled

`nanochat/checkpoint_manager.py` builds `GPTConfig` from saved metadata; add an optional **config override** so we can enable MoE without retraining from scratch.

1. Update `build_model(..., config_overrides=None)`:

```python
def build_model(checkpoint_dir, step, device, phase, config_overrides=None):
    model_data, optimizer_data, meta_data = load_checkpoint(...)
    model_config_kwargs = meta_data["model_config"]
    if config_overrides:
        model_config_kwargs.update(config_overrides)
    model_config = GPTConfig(**model_config_kwargs)
    with torch.device("meta"):
        model = GPT(model_config)
    ...
```

2. Thread the dict through `load_model_from_dir`/`load_model`.

Now `scripts/chat_sft.py` can pass:

```python
config_overrides = dict(
    moe_attn=True,
    moe_num_experts=3,
    moe_k=(32,64,128),
    # ...
)
model, tokenizer, meta = load_model(source, device, phase="train",
                                    model_tag=model_tag, step=step,
                                    config_overrides=config_overrides)
```

(Use the existing load path and training loop; you only modified the config assembly.)

---

## 7) Training scripts (SFT) – minimal deltas

In `scripts/chat_sft.py`:

* **Expose flags** via configurator keys (defaults reasonable):

```python
# add near other hyperparams
use_moe_attn = True
moe_num_experts = 3
moe_k = (32, 64, 128)
moe_router_hidden = 128
moe_router_temp_init = 2.0
moe_router_temp_min = 0.3
moe_router_gumbel = True
moe_router_hard = True
moe_load_balance_lambda = 0.01
moe_distill_lambda = 0.5
```

* When loading the model, pass `config_overrides` as shown above.
* **Phase control**: to do a simple two‑phase run in one file:

  * **Phase 1 (Distill):** Freeze **all** parameters **except** router and `c_proj` (optional) for N steps/epochs. Use higher router temperature.
  * **Phase 2 (Fine‑tune):** Unfreeze all, anneal temperature.

Freezing example:

```python
def set_requires_grad(model, predicate):
    for name, p in model.named_parameters():
        p.requires_grad = predicate(name)

# Distill phase: only router (and optionally attention c_proj) train
set_requires_grad(model, lambda name: "attn.router" in name or "attn.c_proj" in name)

# ...after phase 1
set_requires_grad(model, lambda name: True)  # unfreeze
```

* At each step, optionally adjust router temperature:

```python
temp = max(moe_router_temp_min,
           moe_router_temp_init * (1 - step / num_iterations))
model.set_moe_router_temperature(temp)
```

No change is needed in the loop body (it already does `loss = model(...)`). 

---

## 8) Inference (Engine) – zero or tiny changes

* **No API changes** are required. During inference, the model sees `kv_cache` and `Tq==1` or chunked decode and creates proper per‑expert local masks internally (as in dense). The Engine continues to call:

```python
logits = self.model.forward(ids, kv_cache=kv_cache)
```

* Optionally, set a **lower router temperature** once after loading (e.g., in `scripts/chat_web.py` right after `load_model(...)`):

```python
model.set_moe_router_temperature(model.config.moe_router_temp_min)
```

The Engine’s `KVCache` stays untouched. 

---

## 9) Optional fast path: fused sparse kernel (Triton/CUDA)

After v1 correctness lands, add `nanochat/kernels/triton_sparse_attn.py`:

* Inputs:

  * `Q: (B,H,Tq,D)`, `K: (B,Hkv,Tk,D)`, `V: (B,Hkv,Tk,D)`
  * `indices_ptr` or implicit local windows per token
* Steps:

  1. Compute per‑token **block‑gather** of K/V into shared memory based on indices/window.
  2. Do numerically stable SDPA over gathered keys.
  3. Write back per‑token result.

Guard it behind a flag `--moe_triton=True` and fall back to the PyTorch mask path otherwise.

> Note: PyTorch’s SDPA only hits its fastest path without custom masks. Our boolean mask forces a slower kernel. The fused path restores the O(N·k) memory footprint and speed.

No change to public interfaces; keep it an internal optimization.

---

## 10) Tests

Create `tests/test_moe_attention.py`:

1. **Shape & dtype**: Forward on random inputs with and without KV cache (Tq==Tk, Tq==1, prefix+chunk) → out shape `(B,T,C)`; dtype matches.
2. **GQA parity**: Run with `n_head != n_kv_head` and verify no runtime errors (GQA path is used in dense). 
3. **Router sanity**: `router_probs.sum(-1)≈1`, usage stats roughly uniform with high temperature.
4. **Distillation**: With `moe_distill_lambda>0`, ensure `last_distill_mse` not `None` and loss decreases across a few steps.
5. **Equivalence limit**: If `k_i ≥ Tk` for all experts and soft routing uniform, MoE ≈ dense; MSE(y_moe, y_dense) is small.
6. **KV cache**: Prefill + decode path runs without touching `KVCache` API. 

---

## 11) Logging (W&B)

During SFT/finetuning, log:

* `moe/router_entropy` (mean per token)
* `moe/usage_e{i}` for each expert
* `moe/load_balance` term
* `moe/distill_mse`

This integrates seamlessly with existing logging in `scripts/chat_sft.py` (append to the dict that is logged each step). 

---

## 12) Minimal diffs you will actually write (copy‑paste friendly)

**A) `nanochat/gpt.py` – config fields, Block switch, loss accumulation, temperature setter**

```diff
@@ @dataclass
 class GPTConfig:
   ...
   n_embd: int = 768
+  # --- MoE Attention ---
+  moe_attn: bool = False
+  moe_num_experts: int = 3
+  moe_k: tuple[int, ...] = (32, 64, 128)
+  moe_router_hidden: int = 128
+  moe_router_temp_init: float = 2.0
+  moe_router_temp_min: float = 0.3
+  moe_router_gumbel: bool = True
+  moe_router_hard: bool = True
+  moe_load_balance_lambda: float = 0.01
+  moe_distill_lambda: float = 0.5
+  moe_window_selection: str = "local"
+  moe_topk_per_head: bool = False

@@ class Block(nn.Module):
-    self.attn = CausalSelfAttention(config, layer_idx)
+    if config.moe_attn:
+        from nanochat.moe_attention import AdaptiveAttentionMoE
+        self.attn = AdaptiveAttentionMoE(config, layer_idx)
+    else:
+        self.attn = CausalSelfAttention(config, layer_idx)

@@ class GPT(nn.Module):
     def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
         ...
-        if targets is not None:
-            # training mode: return the loss
-            logits = self.lm_head(x)
-            logits = softcap * torch.tanh(logits / softcap)
-            logits = logits.float()
-            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
-            return loss
+        if targets is not None:
+            logits = self.lm_head(x)
+            logits = softcap * torch.tanh(logits / softcap)
+            logits = logits.float()
+            CE = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
+            # MoE aux losses
+            if getattr(self.config, "moe_attn", False):
+                moe_lb, moe_distill = 0.0, 0.0
+                for block in self.transformer.h:
+                    attn = getattr(block, "attn", None)
+                    if attn is None: continue
+                    if getattr(attn, "last_usage", None) is not None:
+                        usage, E = attn.last_usage, attn.last_usage.numel()
+                        moe_lb = moe_lb + (usage - (1.0 / E)).pow(2).sum()
+                    if getattr(attn, "last_distill_mse", None) is not None:
+                        moe_distill = moe_distill + attn.last_distill_mse
+                CE = CE + self.config.moe_load_balance_lambda * moe_lb \
+                         + self.config.moe_distill_lambda * moe_distill
+            return CE

+    def set_moe_router_temperature(self, t: float):
+        if not getattr(self.config, "moe_attn", False): return
+        for block in self.transformer.h:
+            if hasattr(block.attn, "set_router_temperature"):
+                block.attn.set_router_temperature(t)
```

**B) New file `nanochat/moe_attention.py`** – paste the `Router` and `AdaptiveAttentionMoE` definitions from sections 3–4.

**C) `nanochat/checkpoint_manager.py` – config overrides**

```diff
-def build_model(checkpoint_dir, step, device, phase):
+def build_model(checkpoint_dir, step, device, phase, config_overrides=None):
     ...
-    model_config = GPTConfig(**model_config_kwargs)
+    if config_overrides:
+        model_config_kwargs.update(config_overrides)
+    model_config = GPTConfig(**model_config_kwargs)
     ...
```

Thread the `config_overrides` parameter through `load_model_from_dir`/`load_model` as needed.

**D) `scripts/chat_sft.py` – pass overrides & anneal temperature (minimal)**

Add configurator keys (see §7). Then:

```python
config_overrides = dict(
    moe_attn=use_moe_attn,
    moe_num_experts=moe_num_experts,
    moe_k=moe_k,
    moe_router_hidden=moe_router_hidden,
    moe_router_temp_init=moe_router_temp_init,
    moe_router_temp_min=moe_router_temp_min,
    moe_router_gumbel=moe_router_gumbel,
    moe_router_hard=moe_router_hard,
    moe_load_balance_lambda=moe_load_balance_lambda,
    moe_distill_lambda=moe_distill_lambda,
)
model, tokenizer, meta = load_model(source, device, phase="train",
                                    model_tag=model_tag, step=step,
                                    config_overrides=config_overrides)

# before training loop
model.set_moe_router_temperature(moe_router_temp_init)

# inside loop (after step increments or using lrm schedule)
temp = max(moe_router_temp_min, moe_router_temp_init * (1 - step / num_iterations))
model.set_moe_router_temperature(temp)
```

Training loop remains otherwise unchanged (still does `loss.backward()`, `opt.step()`, logs). 

---

## 13) Performance notes and future upgrades

* **Mask path (v1)**: Simple and correct; creates an `[Tq × Tk]` boolean mask per expert path → higher memory than fused O(N·k). Keep sequences conservative during development.
* **Fused Triton path**: Implement once v1 is stable. Expose `--moe_triton` flag. Kernel fuses *gather + SDPA + reduce* and supports variable `k_i` per token.
* **Cache routing**: In long decoding sessions, caching the hard assignment `idx` (per token, per layer) can avoid recomputing router MLP for repeated tokens. This is a micro‑optimization you can add later.
* **Global tokens**: Add a “reflective” expert with a window equal to full prefix for special positions (e.g., BOS, sentence boundaries) if quality dips on long‑range tasks.

---

## 14) Validation checklist

* **Unit tests** pass (see §10).
* **Quick smoke**: `scripts/chat_sft.py` for 200 steps on a tiny model with MoE enabled should complete and log router usage and distill loss decreasing.
* **Functional parity**: Chat web server responds normally (no API changes).
* **Speed**: With larger contexts (≥4k), per‑step wall‑clock should drop vs dense when you later switch to the fused path; with the mask path you get correctness and a clean upgrade path.

---

## 15) References into current repo (for diffs/context)

* Dense attention’s three control‑flow branches (where to mirror masks / causal logic in MoE): `CausalSelfAttention.forward` using `scaled_dot_product_attention` and the explicit `[Tq × Tk]` “prefix + causal chunk” mask. 
* KV cache class you are reusing as‑is: shape, `insert_kv`, and semantics. 
* Where the training loop expects a single scalar loss (so we add MoE aux terms inside the model): SFT step uses `loss = model(train_inputs, train_targets)` and logs/scales LR around it. 
* Project’s dependency baseline around torch 2.8.0 (Triton optional later). 

---

### Implementation order (bite‑sized PRs)

1. **Config + router + MoE module**, mask path only; keep dense as default, MoE behind a flag.
2. **Model loss plumbing** (load‑balance + distill), no changes to scripts.
3. **Checkpoint overrides** to enable MoE on an existing base/mid model.
4. **SFT flags** + temperature anneal + (optional) freeze/unfreeze phases.
5. **Tests**.
6. **Triton fused kernel** (optional, behind a flag) + perf tests.

Once these land, you’ve got an MoE attention path that plugs directly into nanochat’s Block, respects its KV‑cache/inference behavior, trains with a stable two‑phase recipe, and can be optimized further without touching public APIs.
