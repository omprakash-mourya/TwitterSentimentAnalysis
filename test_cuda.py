import torch

print("🚀 GPU ACCELERATION TEST")
print("=" * 40)
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU memory: {gpu_memory:.1f}GB")
    print(f"GPU compute capability: {torch.cuda.get_device_capability(0)}")
    
    # Test GPU with a simple operation
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.mm(x, y)
    print(f"✅ GPU test successful! Result shape: {z.shape}")
    
    # Clear GPU memory
    del x, y, z
    torch.cuda.empty_cache()
    print("✅ GPU memory cleared")
else:
    print("❌ CUDA not available")
