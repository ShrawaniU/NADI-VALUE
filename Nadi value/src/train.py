from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "data.yaml"
RESULTS = ROOT / "results"

model = YOLO("yolo11n.pt")

model.train(
    data=str(DATA),
    epochs=50,
    imgsz=640,
    batch=4,
    workers=0,
    project=str(RESULTS),
    name="nadi_value_yolo11",
)
