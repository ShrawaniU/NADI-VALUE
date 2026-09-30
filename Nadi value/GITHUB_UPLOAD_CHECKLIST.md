# GitHub Upload Checklist

This archive is repository-root ready. After extracting it, upload the CONTENTS of this folder to the root of your GitHub repository.

Do NOT upload only the ZIP file if you want GitHub reviewers to browse the code and dataset directly.

Required structure:

    dataset/images/train       -> training photos
    dataset/images/val         -> validation photos
    dataset/labels/train       -> YOLO training labels
    dataset/labels/val         -> YOLO validation labels
    dataset/data.yaml          -> dataset configuration
    src/train.py               -> model training
    src/export_onnx.py         -> ONNX export
    src/app_onnx.py            -> deployment application
    src/qnn_provider_check.py  -> QNN provider check
    models/                    -> trained/exported model location
    results/                   -> training output location

The included starter dataset contains 5 training photos + 4 validation photos, with matching YOLO labels. Expand it substantially before claiming meaningful model accuracy.

best.pt and nadi_value.onnx are not included because they must be generated from the custom dataset.
