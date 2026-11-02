from __future__ import annotations

import shutil
import sys


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")

    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA runtime: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        print(f"PyTorch check failed: {exc}")

    try:
        import cv2
        print(f"OpenCV: {cv2.__version__}")
    except Exception as exc:
        print(f"OpenCV check failed: {exc}")

    try:
        import ultralytics
        print(f"Ultralytics: {ultralytics.__version__}")
    except Exception as exc:
        print(f"Ultralytics check failed: {exc}")

    try:
        import easy_dwpose
        print(f"easy-dwpose: {getattr(easy_dwpose, '__version__', 'installed')}")
    except Exception as exc:
        print(f"easy-dwpose check failed: {exc}")

    try:
        import onnxruntime as ort
        print(f"ONNX Runtime: {ort.__version__}")
        print(f"ORT providers: {ort.get_available_providers()}")
        if "CUDAExecutionProvider" in ort.get_available_providers():
            print("CUDAExecutionProvider available.")
        else:
            print("CPU mode: CUDAExecutionProvider unavailable; this is expected on non-NVIDIA systems.")
    except Exception as exc:
        print(f"ONNX Runtime check failed: {exc}")

    ffmpeg = shutil.which("ffmpeg")
    print(f"ffmpeg: {ffmpeg or 'NOT FOUND'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
