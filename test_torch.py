import torch
print(torch.__version__)  # Should output "2.6.0" or similar
print(torch.backends.mps.is_available())  # Should be "True" for M1/M2 support