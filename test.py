import torch

t = 0.1

a = torch.tensor([[1, 2, 3],
                 [4, 5, 6]])

b = torch.tensor([[10, 20, 30],[40, 50, 60]])


print(a / b)
