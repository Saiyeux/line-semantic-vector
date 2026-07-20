# Line Semantic Vector

Lightweight tools for line-art semantic masks and vector debug previews.

This is a slim fork derived from
[ivanpuhachov/line-drawing-vectorization-polyvector-flow](https://github.com/ivanpuhachov/line-drawing-vectorization-polyvector-flow).

This repository intentionally avoids the original C++/Qt/Gurobi vectorization
stack. The current workflow uses Python only:

- split a 512-style portrait line drawing into semantic ink masks
- extract approximate stroke paths and export a debug PNG plus SVG
- optionally predict keypoint anchors with the bundled PyTorch model

## Directory Layout

```text
line-semantic-vector/
├── inputs/                  # local input images, ignored by git
├── outputs/                 # generated masks, previews, SVGs, ignored by git
├── prediction/              # optional keypoint model
│   ├── best_model_checkpoint.pth
│   ├── layers.py
│   ├── sketchyimage.py
│   └── usemodel.py
└── tools/
    ├── semantic_masks_preview.py
    └── vector_debug_preview.py
```

## Requirements

The scripts have been tested with conda base at `/opt/anaconda3/bin/python`.

Required for mask and vector preview:

```bash
opencv-python
numpy
scikit-image
```

Required only for keypoint prediction:

```bash
torch
matplotlib
Pillow
```

## Semantic Masks

Generate semantic masks from a line-art portrait:

```bash
/opt/anaconda3/bin/python tools/semantic_masks_preview.py \
  --input inputs/your_lineart.png \
  --out-dir outputs/your_lineart_masks
```

Outputs include:

```text
semantic_overlay.png
semantic_masks.json
eyes_mask.png
glasses_mask.png
brows_mask.png
nose_mask.png
mouth_mask.png
ears_mask.png
hair_mask.png
face_outline_mask.png
neck_clothes_mask.png
remainder_mask.png
```

Current semantic regions are tuned for frontal portrait line art at roughly
512x512 composition. The scripts scale the region templates to the input image
size, but large pose or crop changes will need retuning.

## Vector Debug Preview

Generate an orange stroke preview and SVG from a line-art image:

```bash
/opt/anaconda3/bin/python tools/vector_debug_preview.py \
  --input inputs/your_lineart.png \
  --png outputs/your_lineart_vector_debug.png \
  --svg outputs/your_lineart_vector_debug.svg
```

If you have predicted keypoints, pass them with `--points`:

```bash
/opt/anaconda3/bin/python tools/vector_debug_preview.py \
  --input inputs/your_lineart.png \
  --points outputs/your_lineart_auto.pts \
  --png outputs/your_lineart_vector_debug.png \
  --svg outputs/your_lineart_vector_debug.svg
```

The vector preview treats dense filled ink islands, such as pupils or heavy
eyebrows, as outline paths instead of skeletonizing them into noisy medial-axis
branches.

## Optional Keypoint Prediction

Predict keypoint anchors:

```bash
MPLCONFIGDIR=/tmp/mplconfig /opt/anaconda3/bin/python prediction/usemodel.py \
  --model prediction/best_model_checkpoint.pth \
  --input inputs/your_lineart.png \
  --output outputs/your_lineart_auto.pts
```

The keypoint model was trained for line drawings. It is not suitable for raw
photos without first converting the photo into clean line art.

## Example End-to-End

```bash
mkdir -p inputs outputs

MPLCONFIGDIR=/tmp/mplconfig /opt/anaconda3/bin/python prediction/usemodel.py \
  --model prediction/best_model_checkpoint.pth \
  --input inputs/portrait_lineart.png \
  --output outputs/portrait_lineart_auto.pts

/opt/anaconda3/bin/python tools/semantic_masks_preview.py \
  --input inputs/portrait_lineart.png \
  --out-dir outputs/portrait_lineart_masks

/opt/anaconda3/bin/python tools/vector_debug_preview.py \
  --input inputs/portrait_lineart.png \
  --points outputs/portrait_lineart_auto.pts \
  --png outputs/portrait_lineart_vector_debug.png \
  --svg outputs/portrait_lineart_vector_debug.svg
```
