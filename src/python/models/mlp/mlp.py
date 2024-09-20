import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use device in your model
#model.to(device)
