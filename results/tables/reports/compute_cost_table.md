# GPU / parameter / inference cost table

Date: 2026-09-23. Hardware: Apple Silicon, PyTorch MPS backend, single GPU,
128×128 grayscale. Training times are wall-clock, sharing the GPU with 0-2
other jobs for some entries (noted); evaluation timings below marked "solo"
ran without contention. All numbers are measured, not estimated, except
where marked.

## Training cost

| Model | Parameters | Epochs run | Wall-clock | Contention |
|---|---:|---:|---:|---|
| Gated conv, deterministic (SOCOFing base) | 5,679,009 | 50 | 30.0 min | solo |
| Gated conv, NIST302 fine-tune (support+spectrum, uniform-confidence control) | 5,679,009 | 18/20 | 32.1 min | solo |
| Gated conv, NIST302 fine-tune (support+spectrum, calibrated confidence) | 5,679,009 | 13/20 (interrupted, never resumed — script has no resume path) | not recorded | 2 concurrent jobs |
| Visible-support predictor (`FingerprintSupportPredictor`) | not logged | 12 | 9.9 min | solo |
| CVAE, spatial, bilinear | 4,898,129 | 20 | 17.9 min | solo |
| DDPM, pixel-space, conditional | 8,280,897 | 15/25 (early-stopped) | 16.3 min | solo |
| DDPM, residual conditional (this session's new model) | 8,281,377 (denoiser only; +5.7M frozen coarse model) | 17/25 (early-stopped) | 30.2 min | 2 concurrent jobs |

The residual model's near-doubled wall-clock at an identical parameter count
to the pixel model is GPU contention (it trained alongside the uniform-
confidence fine-tune and RePaint evaluation), not a per-step cost difference —
its per-epoch cost should be re-measured solo before citing a clean number.

## Inference / sampling cost (full validation, 287 images)

| Sampler | K | Steps | Wall-clock | Sec/image | Sec/sample |
|---|---:|---:|---:|---:|---:|
| DDIM | 10 (max of K=5,10 requested) | 20 | 3.0 min | 0.626 | 0.063 |
| Residual DDPM + DDIM | 10 | 20 | 5.3 min | 1.104 | 0.110 |
| RePaint (ancestral, resampling=2) | 5 | 500×2 refinements | 92.5 min | 19.35 | 3.87 |

RePaint costs roughly **35-60x more per sample** than DDIM-20 for this
configuration, consistent with running the full 500-step ancestral schedule
with 2x refinement at every step versus DDIM's 20 selected steps. It was
evaluated at K=5 instead of K=10/50 specifically because of this cost; a
K=50 RePaint run at the current per-sample cost would take approximately
16 hours and was not attempted.

## Practical implications for the write-up

- **RePaint's fidelity win (Section 5 of the ablation table) has a real
  latency cost.** Any claim that RePaint is "the best model" should be
  paired with this ~35-60x inference cost, not stated unconditionally.
- **The residual DDPM's fidelity gain is nearly free relative to plain DDIM**
  (0.11 vs 0.06 sec/sample, both cheap) — its cost is entirely upstream, in
  needing a trained, frozen coarse deterministic model as a prerequisite,
  which is itself cheap (~30 min) and already required for support
  conditioning elsewhere in the pipeline.
- **No VRAM/peak-memory figures are recorded** for any run in this project;
  `torch.mps` does not expose the same peak-allocation API as CUDA, and no
  measurement was added this session. This is a genuine gap, not an
  oversight to gloss over — if the final report needs a memory figure, it
  requires new instrumentation (e.g. periodic `torch.mps.current_allocated_memory()`
  polling during a run), not a re-read of existing logs.
