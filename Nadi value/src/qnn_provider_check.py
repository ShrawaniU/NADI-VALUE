try:
    import onnxruntime as ort
except ImportError:
    raise SystemExit("Install ONNX Runtime first: pip install onnxruntime")

providers = ort.get_available_providers()

print("Available ONNX Runtime providers:")
for p in providers:
    print(" -", p)

if "QNNExecutionProvider" in providers:
    print("\nQNNExecutionProvider is available.")
    print("The runtime can use Qualcomm's QNN path when the model and environment are configured correctly.")
else:
    print("\nQNNExecutionProvider is NOT available in this Python environment.")
    print("This does not prove the PC has no NPU; it means this runtime is not exposing QNN.")
