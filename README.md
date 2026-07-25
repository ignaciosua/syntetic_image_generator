# invertible_generator

Six deterministic synthetic-data generators (image, stereo image, audio, binaural,
video, stereo video) driven by the same parameter vector, plus a shared-latent
autoencoder that learns to reconstruct each modality and cross-decode between them.

Full history, decisions, and current numbers: see `PROJECT.md`. This file is
just the map of where things live.

## Layout

Every runnable script lives 1 level deep and is meant to be invoked **from the repo
root** (`python training/train_solo.py`, not `cd training && python train_solo.py`).
Each subfolder's scripts add `generators/`, `model/`, `training/`, `tools/` to
`sys.path` on import, so cross-folder imports (`from mel import log_mel`,
`from encoders_multimodal import build_encoders`, etc.) resolve regardless of which
folder the importing file lives in.

```
generators/     the 6 content generators + shared rendering helpers
  synthetic_*_generator.py   deterministic, idx -> content (image/audio/video + stereo)
  levels.py                  level table shared by all generators
  wave.py                    plane-wave-sum content family (levels 148-153), Sobol-sampled
  scene.py / stereo.py / mel.py   scene attrs, stereo geometry, log-mel

model/          the shared-latent architecture itself
  config_multimodal.py       hyperparameters for the current 6-modality system
  dataset_multimodal.py      cached dataset loader, all 6 modalities
  encoders_multimodal.py     per-modality encoders -> shared LATENT_DIM
  decoders_multimodal.py     per-modality decoders <- shared LATENT_DIM
  blocks.py                  shared ResBlock2d/3d used by the encoders/decoders

training/       the two training entry points
  train_solo.py               train ONE modality's autoencoder alone (current phase:
                               verify every modality reconstructs well before touching
                               cross-modal training)
  train_multimodal.py         shared-latent training: self-recon + cross-modal pairs

tools/          everything that reads a checkpoint or a generator but doesn't train
  eval_multimodal.py           retrieval / reconstruction metrics on a checkpoint
  demo_multimodal.py           cross-modal demo (encode A, decode B)
  demo_reconstruction.py       self-reconstruction demo (held-out, per modality)
  audit_generators.py          checks the generators themselves (determinism, coverage)
  audit_pairing.py             checks cross-modality correlation / pairing quality
  probe_wave.py                 checks whether wave-family parameters are recoverable
  ceiling_test.py               isolates one architecture variable at a time
  genparams.py                  classic-level parameter recoverability probe
  precompute_all.py             parallel cache generation, all 6 modalities

checkpoints/                 gitignored. best_multimodal.pt (current shared-latent
                              best), solo/ (per-modality solo runs), legacy_ae/
                              (pre-multimodal system, see legacy_single_modality/)
dataset_cache/                gitignored. precomputed .npy per modality/split
demo_output/                  gitignored. output of demo_multimodal.py / demo_reconstruction.py

media/                        tracked. one-off validation renders from past sessions
logs/                         tracked. training/precompute logs from past runs
legacy_single_modality/        pre-multimodal system (single image generator + CNN
                              retrieval encoder) — superseded but kept for reference,
                              see its own README
```

## Environment

```bash
python  # conda `base` env: PyTorch 2.10, CUDA 12.8 — verified working for this repo
```

## Disk

`dataset_cache/` and `checkpoints/` are the only things that grow unbounded.
Check `df -h` before generating more data — video/stereo_video dominate cache size
(32 frames × 32×32×3 per sample, ~30x an image).
