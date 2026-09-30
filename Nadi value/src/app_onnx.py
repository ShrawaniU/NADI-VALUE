import os
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
from tkinter import Tk, filedialog

MODEL = Path("models/nadi_value.onnx")
IMG_SIZE = 640
CONF = 0.35

NAMES = ["idol", "plastic", "cloth", "flower"]
RECOVERY = {
    "idol": ("RECOVER", "Safe/material recovery and reuse pathway"),
    "plastic": ("RECYCLE", "Plastic recycling pathway"),
    "cloth": ("REUSE / RECYCLE", "Textile recovery pathway"),
    "flower": ("COMPOST", "Organic/composting pathway"),
}

def choose_providers():
    available = ort.get_available_providers()
    if "QNNExecutionProvider" in available:
        return ["QNNExecutionProvider", "CPUExecutionProvider"], "QNN"
    return ["CPUExecutionProvider"], "CPU"

def letterbox(img, size=640):
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    x = (size - nw) // 2
    y = (size - nh) // 2
    canvas[y:y+nh, x:x+nw] = resized
    return canvas, scale, x, y

def preprocess(img):
    canvas, scale, pad_x, pad_y = letterbox(img, IMG_SIZE)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None]
    return x, scale, pad_x, pad_y

def postprocess(output, original, scale, pad_x, pad_y):
    """
    Handles common YOLO11 ONNX output:
    [1, 4+classes, N]
    where boxes are cx,cy,w,h and class scores.
    """
    pred = output
    if isinstance(pred, list):
        pred = pred[0]
    pred = np.asarray(pred)

    if pred.ndim == 3:
        pred = pred[0]

    # Convert [C,N] -> [N,C] if needed.
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T

    if pred.shape[1] < 5:
        return []

    boxes_xywh = pred[:, :4]
    scores = pred[:, 4:]
    class_ids = np.argmax(scores, axis=1)
    confs = np.max(scores, axis=1)

    results = []
    h, w = original.shape[:2]

    for box, cls_id, conf in zip(boxes_xywh, class_ids, confs):
        if float(conf) < CONF or int(cls_id) >= len(NAMES):
            continue

        cx, cy, bw, bh = map(float, box)

        # Undo letterbox.
        cx = (cx - pad_x) / scale
        cy = (cy - pad_y) / scale
        bw /= scale
        bh /= scale

        x1 = max(0, int(cx - bw / 2))
        y1 = max(0, int(cy - bh / 2))
        x2 = min(w - 1, int(cx + bw / 2))
        y2 = min(h - 1, int(cy + bh / 2))

        name = NAMES[int(cls_id)]
        action, detail = RECOVERY[name]

        results.append({
            "name": name,
            "confidence": float(conf),
            "box": (x1, y1, x2, y2),
            "action": action,
            "detail": detail,
        })

    return results

def main():
    if not MODEL.exists():
        raise SystemExit(
            f"Missing {MODEL}. Export the trained custom YOLO11 model first."
        )

    providers, execution = choose_providers()
    print("Execution path:", execution)
    print("Providers:", providers)

    session = ort.InferenceSession(str(MODEL), providers=providers)
    print("Session providers:", session.get_providers())

    root = Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select river-waste image",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")],
    )
    root.destroy()

    if not path:
        print("No image selected.")
        return

    image = cv2.imread(path)
    if image is None:
        raise SystemExit("Could not read image.")

    x, scale, pad_x, pad_y = preprocess(image)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: x})

    detections = postprocess(outputs[0], image, scale, pad_x, pad_y)

    for d in detections:
        x1, y1, x2, y2 = d["box"]
        label = f'{d["name"]} {d["confidence"]:.0%}'
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image, label, (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2
        )
        print(
            f'{d["name"]}: {d["confidence"]:.1%} | '
            f'{d["action"]} | {d["detail"]}'
        )

    cv2.putText(
        image,
        f"NADI-VALUE | {execution} inference",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )

    cv2.imshow("NADI-VALUE", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
