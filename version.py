import torch
print("PyTorch Version:", torch.__version__) 
print("CUDA Version:", torch.version.cuda)
print("cuDNN Version:", torch.backends.cudnn.version())
