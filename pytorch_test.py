from __future__ import print_function
import torch
x = torch.rand(5, 3)
print(x)

print("is cuda available: " + str(torch.cuda.is_available()))
