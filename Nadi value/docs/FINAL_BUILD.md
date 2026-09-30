# Final Build Definition

## What the judge can see

A Windows desktop application that accepts a river/immersion image and produces:

1. object bounding boxes
2. detected waste category
3. confidence
4. recovery action

Example:

plastic → RECYCLE
flower → COMPOST
cloth → REUSE / RECYCLE
idol → RECOVER

## Why Snapdragon

The AI detector is exported to ONNX and the application selects Qualcomm's QNN Execution Provider when it is available. This creates a local inference path suitable for a Snapdragon Windows PC and avoids sending camera frames to a cloud service.

Qualcomm AI Hub currently provides YOLOv11-Detection deployment support for Snapdragon X Elite, X Plus 8-Core and X2 Elite reference devices.

## What is actually demonstrated

Development/training:
- conventional Windows laptop

Target deployment:
- Snapdragon-powered HP Windows 11 PC

The target PC must have the appropriate QNN-capable runtime/environment. The application reports its active execution provider so the demo does not silently pretend to use the NPU.

## Demo story

1. Open a photo of a river/visarjan scene.
2. NADI-VALUE detects the waste.
3. The system assigns a recovery pathway.
4. Explain that the goal is not only cleanup but material recovery and value creation.
5. Explain that inference is intended to run locally on Snapdragon hardware.
