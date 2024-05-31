import torch
import time

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Create large tensors
tensor_size = (10000, 10000)
tensor_cpu = torch.randn(tensor_size)
tensor_gpu = tensor_cpu.to(device)

# Matrix multiplication on CPU
start_time = time.time()
result_cpu = torch.matmul(tensor_cpu, tensor_cpu)
end_time = time.time()
cpu_time = end_time - start_time
print(f"CPU time: {cpu_time:.6f} seconds")

# Matrix multiplication on GPU (if available)
if device.type == 'cuda':
    start_time = time.time()
    result_gpu = torch.matmul(tensor_gpu, tensor_gpu)
    end_time = time.time()
    gpu_time = end_time - start_time
    print(f"GPU time: {gpu_time:.6f} seconds")
    print(f"GPU speedup: {cpu_time / gpu_time:.2f}x")
else:
    print("GPU not available, skipping GPU computation.")