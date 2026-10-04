import torch

# 打印PyTorch版本
print(f"PyTorch版本: {torch.__version__}")

# 检查CUDA是否可用并打印CUDA版本
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"CUDA设备数量: {torch.cuda.device_count()}")
    print(f"当前使用的CUDA设备: {torch.cuda.current_device()}")
    print(f"CUDA设备名称: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA不可用")