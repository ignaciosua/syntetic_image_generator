# invertible_generator

Retrieval autoencoder for the deterministic synthetic image generator.

## Problem

We have `make_image(idx) → (32,32,3)` — deterministic, 124 levels, ~50k images.
We want `encode(image) → idx` so we can:

1. **Reconstruct** in-dataset images exactly: `make_image(encode(image)) ≈ image`
2. **Approximate** out-of-distribution images: map any photo to its closest synthetic match, recovering parameters (level, geometry, color, lighting, etc.)

## Theory

- Generator is 100% deterministic: same idx always produces the same image
- 97.9% of idx produce byte-unique images (2.1% collisions are identical → harmless)
- L2 distance between samples: 5.68–14.73 (well separated in pixel space)
- Contrastive retrieval with prototypes can learn the inverse mapping

## Architecture

```
Image → CNN Encoder → 256-dim embedding → Cosine × Temperature → Softmax → idx → make_image(idx)
```

See `PROMPT.md` for full technical specification.

## Files

| File | Purpose |
|------|---------|
| `synthetic_image_generator.py` | Deterministic generator (unchanged) |
| `PROMPT.md` | Full technical spec for implementation |
| `encoder.py` | CNN encoder (to be written) |
| `train_retrieval.py` | Training loop (to be written) |
| `config.py` | Hyperparameters (to be written) |

## Environment

```bash
/home/neo/miniconda3/envs/bitnet/bin/python3  # PyTorch 2.10, CUDA 12.8
```
