Below is a clean, minimal “TRM-style” recursive‑reasoning add‑on you can graft onto nanochat without disturbing its lovely simplicity. We’ll:

1. add a tiny refiner that keeps a separate latent **z** and iteratively improves a copy of the model’s last hidden state **y** (inspired by the PDF’s loop “update z from (x,y,z), then update y from (y,z), repeat”; we’ll call each inner pass a **latent recursion**),
2. run a short **deep recursion** of those latent recursions but only backprop through the last one (cheap but effective), and
3. wire it in *after* the Transformer trunk and *before* the lm_head.

This preserves all training/inference surfaces and keeps the code changes small.

> **Mapping to the paper idea (plain English):** Treat the Transformer trunk output as fixed “evidence” **x** for this forward pass. Make a working copy **y = x** and a learnable latent state **z** (initialized to zeros). Do `n` tiny inner updates that first refine **z = f_z([x,y,z])**, then refine **y = y + f_y([y,z])`. Repeat a few times (**T**) where only the *last* outer repeat carries gradient; the first T−1 repeats use detached copies so we get the benefits of recursion without blowing up memory/compute. That’s all.

---

## 0) Files we’ll touch

* **new**: `nanochat/trm.py` – the tiny, self‑contained recursive refiner module
* **edit**: `nanochat/gpt.py` – config fields, instantiate/use the refiner, include its params in the optimizer groups
* **edit**: `scripts/base_train.py` – expose 4 flags and pass them into `GPTConfig` so you can train with/without TRM

Everything else (mid‑train, SFT, web, engine) automatically works because we keep the model I/O unchanged: forward still returns the same logits/loss, generate still calls forward.

---

## 1) Add the TRM refiner (new file)

Create `nanochat/trm.py`:

```python
# nanochat/trm.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TRMRefiner(nn.Module):
    """
    Minimal recursive refiner in the spirit of 'Less is more: Recursive Reasoning with Tiny Networks'.

    x : (B, T, C)  fixed context features from Transformer trunk for this forward pass
    y : (B, T, C)  running "answer" features we iteratively refine (start as y = x)
    z : (B, T, Dz) private latent state we iteratively refine

    Inner update ("latent recursion"):
      z <- f_z([x, y, z]) via a small MLP
      y <- y + f_y([y, z]) via a small MLP (residual update)

    Deep recursion:
      Repeat the inner update n times inside each outer step.
      Do T outer steps; detach (no gradients) for the first T-1 to keep training cheap, backprop only through the last.
    """

    def __init__(self, d_model: int, d_latent: int, n: int = 6, T: int = 3):
        super().__init__()
        self.d_model = d_model
        self.d_latent = d_latent
        self.n = n  # latent-recursion iterations per outer step
        self.T = T  # deep-recursion outer steps (backprop only through the last)

        # z-updater: f_z([x, y, z]) -> z
        self.fz = nn.Sequential(
            nn.Linear(d_model + d_model + d_latent, 4 * d_latent, bias=False),
            nn.SiLU(),
            nn.Linear(4 * d_latent, d_latent, bias=False),
        )

        # y-updater: f_y([y, z]) -> delta_y
        self.fy = nn.Sequential(
            nn.Linear(d_model + d_latent, 4 * d_model, bias=False),
            nn.SiLU(),
            nn.Linear(4 * d_model, d_model, bias=False),
        )

    def _latent_recursion(self, x, y, z):
        # run the small z-updates n times; then do ONE y update (cheap & stable)
        for _ in range(self.n):
            z = self.fz(torch.cat([x, y, z], dim=-1))
        y = y + self.fy(torch.cat([y, z], dim=-1))
        return y, z

    def forward(self, x):
        """
        Args:
            x: final trunk activations (B, T, C) before lm_head
        Returns:
            y: refined activations (B, T, C)
        """
        B, T, C = x.shape
        device, dtype = x.device, x.dtype
        # initialize y,z
        y = x
        z = torch.zeros(B, T, self.d_latent, device=device, dtype=dtype)

        # Deep recursion: detach the first T-1 outer steps
        for t in range(self.T):
            if t < self.T - 1:
                y_det, z_det = y.detach(), z.detach()
                y, z = self._latent_recursion(x, y_det, z_det)
            else:
                y, z = self._latent_recursion(x, y, z)

        return y
```

* This module is intentionally tiny and follows the “z from (x,y,z) then y from (y,z)” cadence from the PDF.
* We default to `n=6` inner updates and `T=3` outer steps (the paper uses small numbers like these).
* Using `detach()` on the first `T-1` outer steps implements the paper’s “only the last step carries gradients”, which keeps the training footprint small while still granting iterative computation.

---

## 2) Wire it into the model

Open `nanochat/gpt.py` and make three surgical edits:

### 2.1 Add config fields

In `GPTConfig`, add four fields with safe defaults so nothing changes unless you turn it on:

```python
# nanochat/gpt.py  (inside @dataclass GPTConfig)
    # --- TRM-style recursive refiner (off by default) ---
    use_trm: bool = False
    trm_d_latent: int = 256
    trm_n: int = 6
    trm_T: int = 3
```

> This extends the simple dataclass; the repo already leans into super-minimal config objects, so this fits the vibe.

### 2.2 Instantiate and run the refiner

Add the import near the other imports:

```python
from nanochat.trm import TRMRefiner
```

Create the refiner in `GPT.__init__` **after** building the trunk/embeddings but **before** `lm_head` usage; for example right after the existing modules are constructed:

```python
# inside GPT.__init__(...)
        # ... existing modules already created above ...
        self.trm = None
        if getattr(config, "use_trm", False):
            self.trm = TRMRefiner(
                d_model=config.n_embd,
                d_latent=config.trm_d_latent,
                n=config.trm_n,
                T=config.trm_T,
            )
```

Now, call the refiner in `forward(...)` **right after** the final `norm(x)` and **before** `lm_head(x)`.

You’ll find the trunk forward and the final norm in the existing code:

> The model builds `x = self.transformer.wte(idx)`, normalizes, loops over blocks, and normalizes again right before logits are computed. That’s our hook point. 

Patch the forward accordingly:

```python
# nanochat/gpt.py  inside GPT.forward(...)
        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx)
        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)
        x = norm(x)

        # --- TRM hook: refine hidden states before the unembedding ---
        if self.trm is not None:
            x = self.trm(x)

        # Compute logits and (optionally) loss as before
        lm_logits = self.lm_head(x)
        # ... rest of the function unchanged ...
```

> We do not alter shapes, targets handling, caching, or generation logic. The refiner is a drop‑in post‑trunk pre‑head adapter, so all public APIs keep working exactly as before.

### 2.3 Make sure the refiner’s params get optimized

nanochat splits parameters into three groups (matrix/embedding/lm_head) and puts linear layers into Muon, embeddings and unembedding into AdamW. The current code collects linear (matrix) params only from the trunk layers:

```python
matrix_params = list(self.transformer.h.parameters())
embedding_params = list(self.transformer.wte.parameters())
lm_head_params = list(self.lm_head.parameters())
assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params)
```

That assert will fail once we add TRM unless we include it. Update `setup_optimizers` to include the refiner in the matrix group:

```python
# nanochat/gpt.py  inside GPT.setup_optimizers(...)
        # Separate out all parameters into 3 groups (matrix, embedding, lm_head)
        matrix_params = list(self.transformer.h.parameters())
        if getattr(self, "trm", None) is not None:
            matrix_params += list(self.trm.parameters())  # <-- add TRM to Muon group
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params)
```

This mirrors the repo’s intended split and keeps the learning‑rate scaling behavior intact. (See the original grouping and LR scaling notes where the optimizer is set up. )

> **Note on FLOPs accounting:** `estimate_flops()` won’t include the TRM headroom. That’s fine by default (the scripts don’t rely on it unless you aim for a FLOPs target), but if you *do* use `target_flops`, expect a slight undercount. The quickest fix is to set `num_iterations` explicitly.

---

## 3) Expose 4 flags in base training

Base training is where we instantiate `GPTConfig`. We’ll thread through minimal flags and pass them into the config. Open `scripts/base_train.py`.

### 3.1 Add four top‑level flags (right under the other “User settings”)

```python
# scripts/base_train.py  (top, under "User settings")
# --- TRM toggles (off by default) ---
use_trm = 0           # 0/1 for False/True
trm_d_latent = 256
trm_n = 6
trm_T = 3
```

These automatically become CLI‑overrideable via nanochat’s little configurator (any top‑level var is fair game), so `--use_trm=1` will work. (The script builds `config_keys` from globals and lets the configurator override them. )

### 3.2 Pass the flags into `GPTConfig`

Find the place where the model config dict is built:

```python
# scripts/base_train.py
model_config_kwargs = dict(sequence_len=max_seq_len, vocab_size=vocab_size, n_layer=num_layers, n_head=num_heads, n_kv_head=num_kv_heads, n_embd=model_dim)
```

Patch it to include the new flags:

```python
model_config_kwargs = dict(
    sequence_len=max_seq_len,
    vocab_size=vocab_size,
    n_layer=num_layers,
    n_head=num_heads,
    n_kv_head=num_kv_heads,
    n_embd=model_dim,
    # --- TRM ---
    use_trm=bool(use_trm),
    trm_d_latent=trm_d_latent,
    trm_n=trm_n,
    trm_T=trm_T,
)
```

The rest of the script remains unchanged — it creates the model on meta device, compiles, estimates FLOPs, sets up optimizers, and trains. (See where `model_config_kwargs` is used to build the model. )

> Mid‑train and SFT don’t need edits: they load the model from the checkpoint’s stored config (we didn’t change loading), so the TRM settings flow through transparently. The loader reconstructs `GPTConfig` exactly from saved metadata before loading weights. 

---

## 4) Sanity check: what did we *not* change?

* **Forward signature**: unchanged; you still call `model(x, y)` for loss, or `model(x)` for logits.
* **kv_cache/generation**: untouched; the refiner sits after the trunk, so caching and rotary embeddings remain exactly as they were. (Your trunk + final norm block is unchanged; we only inserted a post‑norm adapter before `lm_head` where the code already computes logits. )
* **Optimizer regimes**: we simply appended TRM params to the “matrix” (Muon) bucket; AdamW buckets still contain only embeddings and the unembedding/lm_head as before. (Matches the original split & learning‑rate scaling; see `setup_optimizers` grouping. )
* **Checkpointing**: model config is saved and restored, so TRM toggles/values are persisted and respected across phases. (See how model config is saved and later read to rebuild the model. )

---

## 5) How it actually computes (junior‑friendly mental model)

* The Transformer trunk computes **x**, the per‑token features.
* We make **y = x**. That’s the “current guess” of what the features should be before going into the vocab head.
* We keep a small, separate memory **z** per token position (starts as zeros).
* We run `n` tiny updates that weave **x, y, z** together:

  * new **z** = MLP([x, y, z])
  * new **y** = y + MLP([y, z])
* We do that inner loop a few times, then we do that *outer* loop `T` times, detaching the first `T−1` so only the last one carries gradients. Translation: you get extra “thinking steps” at low training cost.
* The final **y** replaces **x** before `lm_head`, and that’s it.

This is the smallest faithful skeleton of the paper’s idea you can add without complicating nanochat’s flow.

---

## 6) Training & toggling it on

**CPU/MPS demo** (tiny settings, just to exercise code paths):

```bash
python -m scripts.base_train \
  --depth=4 --max_seq_len=512 --device_batch_size=1 \
  --total_batch_size=1024 --num_iterations=50 \
  --use_trm=1 --trm_d_latent=128 --trm_n=4 --trm_T=2
```

**GPU training** (typical base run, enable TRM):

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
  --depth=20 --device_batch_size=32 \
  --use_trm=1 --trm_d_latent=256 --trm_n=6 --trm_T=3
```

Everything else (loss eval, midtrain, SFT, chat web) works the same since nothing about I/O changed. See how base training plumbs `model_config_kwargs` through meta‑init/compile/optimizers and prints the FLOPs/token estimate (that estimate does not include TRM). 

---

## 7) Optional niceties (later, if you’re curious)

* **FLOPs accounting:** If you use `target_flops`, you could add a small extra term in `estimate_flops()` to approximate the TRM cost. Quick‑n‑dirty solution for now: specify `--num_iterations` explicitly.
* **Halting head (q):** The paper sometimes predicts a halting logit from the refined **y**. You can add a tiny linear head and a threshold and early‑stop the inner loop at inference — but that’s beyond “minimal”, so we’ve left it out.
* **Block‑wise TRM:** You can drop this adapter per block (or at middle depth) to trade compute for potentially more gains. Keep the same API so the rest of nanochat remains unchanged.

---

## 8) That’s really it — full patch summary

* New `nanochat/trm.py` with `TRMRefiner` (≈40 lines).
* `GPTConfig` gains `use_trm`, `trm_d_latent`, `trm_n`, `trm_T`.
* `GPT.__init__` conditionally creates `self.trm`; `GPT.forward` calls it after the final `norm` and before `lm_head`. (Hook location is the same place logits are computed today. )
* `setup_optimizers()` adds `self.trm.parameters()` to matrix params (Muon). (Matches the repo’s intended three‑bucket scheme. )
* `scripts/base_train.py` surfaces four flags and threads them into `GPTConfig` at model creation time (the place where `model_config_kwargs` is constructed). 

---

### Why this is “minimal but true to the idea”

* We mirror the paper’s **two‑state** view (y and z) and its **recursive inner loop**.
* We use a short **deep recursion** but only backprop through the last outer step, just like the PDF suggests to keep training cheap.
* We keep nanochat’s excellent ergonomics: one new file, a handful of lines changed, zero API drift, and an off‑by‑default toggle so you can A/B easily.

If you want, the next experiments that make this sing are ablation sweeps over `trm_d_latent`, `trm_n`, and `trm_T`, and then measuring CORE/ChatCORE deltas at constant wall‑clock vs. constant token budget. For the exact patch points referenced above: optimizer param grouping and the final‑norm hook are visible in the repo’s packaged listing around `setup_optimizers(...)` and the end of `GPT.forward(...)`.  
