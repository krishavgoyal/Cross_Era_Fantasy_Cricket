# Intel/CPU Laptop Profile

This profile is for systems without an NVIDIA CUDA-capable GPU.

Install PyTorch for CPU and ONNX Runtime CPU. Run the pipeline with `--device cpu --dwpose-device cpu`.

Expected trade-off: preprocessing is functionally supported but materially slower than an NVIDIA CUDA workstation. Do not install NVIDIA drivers merely to obtain `nvidia-smi` on this machine.
