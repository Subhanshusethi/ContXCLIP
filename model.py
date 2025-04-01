class ReluSIG(nn.Module):
    def __init__(self ,):
        super().__init__()
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        Relumoid = self.gelu(x)*self.sigmoid(torch.square(x))
        return Relumoid

