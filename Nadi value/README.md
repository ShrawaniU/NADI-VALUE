# NADI-VALUE
## AI-Powered River Waste Detection & Recovery Assistant

**DETECT → PREVENT → RECOVER → CREATE VALUE**

NADI-VALUE is a four-class YOLO11 computer-vision prototype that identifies idol, plastic, cloth and flower in river/collection-point images and maps each detection to a recovery action.

### Repository structure

```text
NADI-VALUE/
├── dataset/
│   ├── images/
│   │   ├── train/        # PHOTO DATA FOR MODEL TRAINING
│   │   └── val/          # PHOTO DATA FOR VALIDATION
│   ├── labels/
│   │   ├── train/        # YOLO labels matching train photos
│   │   └── val/          # YOLO labels matching validation photos
│   ├── data.yaml
│   └── README.md
├── src/
│   ├── train.py
│   ├── export_onnx.py
│   ├── app_onnx.py
│   ├── qnn_provider_check.py
│   └── recovery_rules.py
├── models/
├── results/
├── docs/
├── requirements.txt
└── run_snapdragon.bat
```

### Train the model

Run these commands from the repository root:

```bat
pip install -r requirements.txt
python src\train.py
```

The training script now points to the real dataset path:

```text
dataset/data.yaml
```

and the dataset file points to:

```text
dataset/images/train
dataset/images/val
dataset/labels/train
dataset/labels/val
```

After training:

```bat
copy "results\nadi_value_yolo11\weights\best.pt" "models\best.pt"
python src\export_onnx.py
```

Place the exported ONNX file at:

```text
models/nadi_value.onnx
```

### Snapdragon deployment

The ONNX application checks whether the installed ONNX Runtime environment exposes `QNNExecutionProvider`. When QNN is available, it selects the QNN execution path; otherwise it uses CPU.

```bat
python src\qnn_provider_check.py
python src\app_onnx.py
```

The intended target is a compatible Snapdragon-powered HP Windows PC. Do not claim Snapdragon NPU performance unless measured on the target device.

### Important dataset note

The included dataset is intentionally small and is for pipeline/prototype testing, not a trustworthy production accuracy claim. It contains 5 training photos and 4 validation photos, with one YOLO label file matching each photo. Replace the starter images with properly licensed/owned photos and expand the dataset before presenting quantitative accuracy.
