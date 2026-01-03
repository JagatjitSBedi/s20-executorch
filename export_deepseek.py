import torch
from executorch.exir import to_edge
# Dummy model (replace with DeepSeek)
model = torch.nn.Linear(1,1)
edge = to_edge(model)
edge.exported_program().to_executorch("s20_deepseek.pte")
print("S20 PTE exported ✓")
