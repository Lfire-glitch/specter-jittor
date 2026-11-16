"""Step 1: convert HF pytorch_model.bin -> .npz (no torch needed at jittor runtime)."""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, 'models', 'specter', 'pytorch_model.bin')
OUT = os.path.join(ROOT, 'models', 'specter', 'specter.npz')


def main():
    sd = torch.load(BIN, map_location='cpu', weights_only=False)
    arrs = {}
    for k, v in sd.items():
        if not hasattr(v, 'numpy'):
            print(f'[skip non-tensor] {k}')
            continue
        arrs[k] = v.detach().numpy().astype(np.float32)
    np.savez(OUT, **arrs)
    size = os.path.getsize(OUT) / 1e6
    print(f'[ok] {len(arrs)} tensors -> {OUT} ({size:.1f} MB)')
    print(f'[ok] total params: {sum(a.size for a in arrs.values())/1e6:.2f}M')


if __name__ == '__main__':
    main()
