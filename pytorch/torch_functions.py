import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
b = torch.tensor([[7, 8], [9, 10], [11, 12]], dtype=torch.float32)

result = torch.mm(a, b)
print(result)

