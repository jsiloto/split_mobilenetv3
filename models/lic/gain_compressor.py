from compressai.layers import GDN
from torch import nn


class GainCompressor(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = GainCompressorEncoder()
        self.decoder = GainCompressorDecoder()


    def forward(self, x):
        output = self.encoder(x)
        output['y_hat'] = self.decoder(output['y_hat'])
        return output

class GainCompressorEncoder(nn.Module):
    def __init__(self):

        super().__init__()

        N = 128
        M = 192

        self.g_a = nn.Sequential(
            nn.Conv2d(3, N),
            GDN(N),
            nn.Conv2d(N, N),
            GDN(N),
            nn.Conv2d(N, N),
            GDN(N),
            nn.Conv2d(N, M),
        )

    def forward(self, x):
        y = self.g_a(x)
        return y

class GainCompressorDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        N = 128
        M = 192
        self.g_s = nn.Sequential(
            nn.Conv2d(M, N),
            GDN(N, inverse=True),
            nn.Conv2d(N, N),
            GDN(N, inverse=True),
            nn.Conv2d(N, N),
            GDN(N, inverse=True),
            nn.Conv2d(N, 3),
        )

    def forward(self, x):
        y = self.g_s(x)
        return y