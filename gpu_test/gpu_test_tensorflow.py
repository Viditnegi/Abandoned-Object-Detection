import tensorflow as tf
import time

# print(tf.__version__())
# Check if GPU is available
if tf.config.list_physical_devices('GPU'):
    device = '/GPU:0'
    print("GPU is available")
else:
    device = '/CPU:0'
    print("GPU is not available")

# Create large tensors
tensor_size = (10000, 10000)
tensor_cpu = tf.random.normal(tensor_size)
with tf.device(device):
    tensor_gpu = tf.random.normal(tensor_size)

# Matrix multiplication on CPU
with tf.device('/CPU:0'):
    start_time = time.time()
    result_cpu = tf.matmul(tensor_cpu, tensor_cpu)
    end_time = time.time()
    cpu_time = end_time - start_time
    print(f"CPU time: {cpu_time:.6f} seconds")

# Matrix multiplication on GPU (if available)
if device == '/GPU:0':
    with tf.device('/GPU:0'):
        start_time = time.time()
        result_gpu = tf.matmul(tensor_gpu, tensor_gpu)
        end_time = time.time()
        gpu_time = end_time - start_time
        print(f"GPU time: {gpu_time:.6f} seconds")
        print(f"GPU speedup: {cpu_time / gpu_time:.2f}x")
else:
    print("GPU not available, skipping GPU computation.")