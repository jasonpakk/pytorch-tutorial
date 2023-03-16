import torch

x = torch.randn(3, requires_grad=True)
print(x)

# to not use requires_grad:
# x.requires_grad(False)
# x.detach()
# with torch.no_grad()

y = x+2
print(y)

z = y*y*2
# z = z.mean()
print(z)

v = torch.tensor([.1, 1.0, 0.001], dtype=torch.float32)
z.backward(v) # dz/dx
print(x.grad)

weights = torch.ones(4, requires_grad=True)

for epoch in range(2):
  model_output = (weights*3).sum()

  model_output.backward()

  print(weights.grad)

  weights.grad.zero_() # <- important to zero out


