
# AutoBugCam - Real-time Camera Artifact Detector (Master Project)

AutoBugCam is a research-grade tool that localizes common camera pipeline artifacts
(HDR halos/ghosting, demosaic zippering, moire, rolling-shutter skew, flare/veiling
glare, denoise smear, banding, color cast) and produces per-pixel heatmaps plus
per-image labels. It ships with:

- Synthetic artifact generators (fully controllable) to create supervised masks
- Lightweight dual-head model (segmentation + multi-label classification)
- Training loop (PyTorch), metrics, and visualization tools
- CLI demo that overlays artifact heatmaps on images
- ONNX export script for deployment experiments

This is a complete starter you can run locally and extend into a publishable master project.

## Quickstart

```bash
# 1) Create a Python 3.10+ env and install deps
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) Generate a tiny synthetic dataset (images + masks + labels)
python data/generate_synth.py --out data/synth --num 200

# 3) Train a small model
python train/train.py data.root=data/synth train.batch_size=4 train.epochs=5

# 4) Visualize predictions on a folder
python app/demo_cli.py --images data/synth/images --out runs/vis
```

Artifacts will be saved under runs/.

## Project Structure

```
autobugcam/
├─ sims/                  # artifact simulators (HDR, RS, demosaic, flare, etc.)
├─ data/                  # dataset + synthetic data generator
├─ models/                # backbones, heads, UNet-like
├─ train/                 # train loop, losses, configs (Hydra)
├─ eval/                  # metrics + visualization helpers
├─ app/                   # CLI demo for overlays
├─ tools/                 # exporters and utilities
└─ README.md
```

## Research Ideas to Extend
- Add a caption head for natural-language explanations and suggested fixes.
- Distill to a mobile-friendly student; export to CoreML / TensorRT.
- Add temporal model for video artifacts (flicker, ghost trails).
- Run a human study comparing usefulness vs standard NR-IQA baselines.

## License
MIT - attribution appreciated.
