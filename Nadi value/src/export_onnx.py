from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
best = ROOT / "results" / "nadi_value_yolo11" / "weights" / "best.pt"
models = ROOT / "models"
models.mkdir(exist_ok=True)

if not best.exists():
    raise FileNotFoundError(
        f"{best} not found. Train the model first with: python src/train.py"
    )

model = YOLO(str(best))
out = model.export(format="onnx", imgsz=640)
print("ONNX export:", out)
print("Copy the generated ONNX file to models/nadi_value.onnx")
