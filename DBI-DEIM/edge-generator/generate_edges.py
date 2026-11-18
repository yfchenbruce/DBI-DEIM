"""Generate paired edge maps without importing Ultralytics."""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from edge_generator import EdgeFeatureGenerator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='Image or directory (recursive)')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cpu', help='cpu or cuda:0')
    parser.add_argument('--gaussian-ksize', type=int, choices=[1, 3, 5, 7, 9], default=5)
    parser.add_argument('--sobel-ksize', type=int, choices=[3, 5], default=3)
    parser.add_argument('--channels', type=int, choices=[1, 3], default=1)
    parser.add_argument('--binarize', action='store_true')
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    if not source.exists():
        parser.error('Input does not exist')
    root = source if source.is_dir() else source.parent
    if output == root or root in output.parents:
        parser.error('Output must be outside the input directory')
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    files = sorted(p for p in source.rglob('*') if p.suffix.lower() in extensions) if source.is_dir() else [source]
    if not files:
        parser.error('No images found')
    model = EdgeFeatureGenerator(gaussian_ksize=args.gaussian_ksize,
                                 sobel_ksize=args.sobel_ksize,
                                 binarize=args.binarize,
                                 threshold=args.threshold, device=args.device).eval()
    targets = [output / p.relative_to(root).with_suffix('.png') for p in files]
    if len(set(targets)) != len(targets):
        parser.error('Input filenames collide after conversion to PNG')
    failures = []
    written = skipped = 0
    for src, dst in zip(files, targets):
        if dst.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            with Image.open(src) as image:
                rgb = np.array(image.convert('RGB'), dtype=np.float32) / 255.0
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(args.device)
            edge = model(tensor)[0, 0].cpu().numpy()
            pixels = np.rint(np.clip(edge, 0, 1) * 255).astype(np.uint8)
            if args.channels == 3:
                pixels = np.repeat(pixels[..., None], 3, axis=2)
            dst.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pixels).save(dst)
            written += 1
        except Exception as exc:
            failures.append((str(src), str(exc)))
    print(f'Written: {written}; skipped: {skipped}; failed: {len(failures)}')
    for src, reason in failures:
        print(f'{src}: {reason}')
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
