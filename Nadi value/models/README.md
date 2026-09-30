# Models

After training, copy:
results/nadi_value_yolo11/weights/best.pt

to:
models/best.pt

After ONNX export, copy the generated ONNX model to:
models/nadi_value.onnx

Do not commit large trained models to GitHub unless your repository policy permits it. For the challenge, source code and dataset structure are the important reproducible parts.
