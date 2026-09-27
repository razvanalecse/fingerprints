# SOCOFing reconstructibility vs observed fraction r: smooth vs segmented fit

Date: 2026-09-23. Source: `outputs/ddim_50step_k10_validation/per-image-probabilistic-metrics.csv`
(SOCOFing DDIM-50 sampler, K=10, full validation split, 900 images, r ∈
{0.10, ..., 0.80}). Method: `scripts/analyze_reconstructibility_segmented.py`
— a smooth (ordinary linear) model in r is compared against a continuous
piecewise-linear ("broken-stick") model with one breakpoint, grid-searched
over the midpoints between the eight observed r levels, via a nested F-test.
The breakpoint location is reported as a curve-fit statistic, not asserted to
be a mechanistic "critical threshold" — that interpretation, if made at all,
belongs in the discussion section with its own justification.

## Results

| Outcome | Linear slope | Linear R² | Segmented breakpoint r | Slope before | Slope after | Segmented R² | F-test p | Segmented preferred (α=0.05) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| SSIM | +0.218 | 0.445 | 0.45 | +0.188 | +0.250 | 0.447 | 0.067 | **no** |
| Orientation error | −0.382 | 0.217 | 0.35 | −0.664 | −0.247 | 0.230 | 1.4e-4 | **yes** |
| Uncertainty magnitude (mean predictive std) | −0.035 | 0.195 | 0.25 | −0.003 | −0.041 | 0.200 | 0.012 | **yes** |
| Sample diversity (pairwise MAE) | −0.045 | 0.211 | 0.25 | −0.010 | −0.051 | 0.216 | 0.024 | **yes** |

## Reading

Three of four outcomes statistically prefer a segmented fit; SSIM does not
(p=0.067, just above the conventional threshold). This is itself a finding:
reconstruction quality metrics do not all bend at the same point, or bend at
all, as more of the fingerprint becomes visible.

- **Orientation error** has the strongest and most interpretable segment: it
  improves steeply up to r≈0.35, then much more slowly beyond that. Once
  roughly a third of the print is visible, showing more of it keeps helping,
  but far less per additional percent than the first third did.
- **Uncertainty magnitude and diversity** share a similar, weaker pattern
  (breakpoint ≈0.25): both stay nearly flat at very low observed fractions,
  then decline noticeably faster once r exceeds ≈0.25. In other words, the
  model's own expressed uncertainty barely shrinks in the hardest regime
  (r ≤ 0.25) even as more is revealed — plausibly because at very low r the
  reconstruction problem is dominated by regions still far from any observed
  evidence regardless of the exact r value, consistent with the
  distance-to-anchor findings elsewhere in this project.
- **SSIM improves smoothly**, with no statistically supported bend — pixel
  structural similarity apparently accrues value from additional observed
  pixels at a roughly constant rate across the whole 10-80% range studied.

## Caveats

- r takes only eight discrete values in this dataset (the SOCOFing masking
  protocol), so the breakpoint search is over seven midpoints, not a
  continuous domain; breakpoint estimates should not be over-interpreted to
  sub-decile precision.
- This uses the DDIM-50 sampler only; the pattern has not been re-checked
  against RePaint or the other SOCOFing samplers, nor against NIST302.
- No multiple-comparison correction was applied across the four outcomes
  tested here (four F-tests); a Holm correction would raise the SSIM-adjacent
  borderline case further from significance but would not change the
  qualitative conclusion for orientation error, whose p-value has three
  orders of magnitude of margin.
