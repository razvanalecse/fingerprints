import hashlib
import os
from pathlib import Path

import numpy as np
from PIL import Image
from fingerprint_reconstruction.preprocessing.orientation import (
    estimate_foreground_mask, estimate_orientation_field, orientation_error)
from fingerprint_reconstruction.metrics import region_image_metrics, summarize_ridge_topology

dl = Path('/Users/razvanalecse/Downloads')
root = dl / 'generalist_baseline_pack' / 'cases'
completion_glob = os.environ.get('GENERALIST_COMPLETION_GLOB', 'generalist_completion_*.png')
def g(p): return np.asarray(Image.open(p).convert('L').resize((128,128), Image.LANCZOS), dtype=np.float32)/255.
truth = g(root/'case01_rect20'/'ground_truth.png'); fg = estimate_foreground_mask(truth)
tf = estimate_orientation_field(truth, use_foreground_mask=False)
tt = summarize_ridge_topology(truth, fg, tf.theta, tf.coherence, tf.valid)

seen=set(); uniq=[]
for p in sorted(dl.glob(completion_glob)):
    h = hashlib.md5(p.read_bytes()).hexdigest()
    if h not in seen: seen.add(h); uniq.append(p)

print('Scored over the fingerprint foreground, against the single case01 ground truth')
print('%-10s %6s %6s %8s %10s %10s' % ('output','MAE','SSIM','orient','minutiae','coherence'))
print('%-10s %6s %6s %8s %10.1f %10.3f' % ('TRUTH','-','-','-', tt.minutiae_density_per_1000px, tt.orientation_coherence_mean))
oes=[]
for p in uniq:
    a = g(p); m = region_image_metrics(truth, a, fg)
    af = estimate_orientation_field(a, use_foreground_mask=False)
    oe = orientation_error(tf, af, region_mask=fg); oes.append(oe)
    st = summarize_ridge_topology(a, fg, af.theta, af.coherence, af.valid)
    print('%-10s %6.3f %6.3f %8.3f %10.1f %10.3f' % (
        p.name[-12:-4], m['mae'], m['ssim_map_mean'], oe,
        st.minutiae_density_per_1000px, st.orientation_coherence_mean))
print()
print('mean orientation error of generalist completions : %.3f' % np.mean(oes))
print("this project's gated model, same 25-case pack     : 0.094")
