import torch.nn as nn
import torch


torch.manual_seed(123)
dropout = nn.Dropout(0.5)
example = torch.ones(6,6)

print(dropout(example))