# Deterministic baseline results

All results use the same 900-image subject-disjoint SOCOFing validation split.
Metrics are computed on missing pixels intersected with the estimated
fingerprint foreground ROI. The test split remains locked.

| Model | Parameters | MAE ↓ | PSNR (dB) ↑ | local SSIM ↑ | axial orientation error ↓ |
|---|---:|---:|---:|---:|---:|
| Nearest-observed interpolation | 0 | 0.27686 | 9.0414 | 0.09412 | 0.81524 |
| U-Net | 7,762,753 | 0.17582 | 12.9664 | 0.25348 | 0.24069 |
| Gated convolution | 5,679,009 | 0.17401 | 13.0216 | 0.25853 | 0.22889 |

## Paired Gated convolution versus U-Net

Improvement is coded so that positive values favour gated convolution.

| Outcome | Mean paired improvement | 95% CI | Cohen dz | Holm-adjusted Wilcoxon p |
|---|---:|---:|---:|---:|
| MAE | 0.00181 | [0.00131, 0.00231] | 0.235 | 7.92e-18 |
| PSNR | 0.05526 dB | [0.02782, 0.08270] | 0.132 | 2.45e-05 |
| local SSIM | 0.00506 | [0.00398, 0.00613] | 0.308 | 5.50e-22 |
| orientation error | 0.01180 | [0.00836, 0.01523] | 0.225 | 1.59e-13 |

The improvement is statistically detectable because the comparison is paired
and contains 900 cases, but its magnitude is small. Visual outputs from both
learned deterministic models remain overly smooth for severely incomplete
inputs. These results justify retaining gated convolution as the stronger
deterministic comparator; they do not establish accurate recovery of true
missing ridge phase.

Observed-pixel MAE is exactly zero for all methods that apply hard data
consistency. This is a constraint, not a learned-performance result.
