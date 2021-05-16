class Fourier(nn.Module):
	def __init__(self, nmb=256, scale=10):
		super(Fourier, self).__init__()
		self.b = torch.randn(2, nmb)*scale
		self.pi = 3.14159265359
	def forward(self, v):
		x_proj = torch.matmul(2*self.pi*v, self.b)
		return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], -1)
