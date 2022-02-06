from torch import nn as nn


class Encoder(nn.Module):
    def __init__(
        self,
        dim_h,
        n_channel,
        n_z,
        **kwargs
    ):
        super(Encoder, self).__init__()

        self.n_channel = n_channel
        self.dim_h = dim_h
        self.n_z = n_z

        # convolutional filters, work excellent with image data
        self.conv = nn.Sequential(
            nn.Conv2d(self.n_channel, self.dim_h, 4, 2, 1, bias=False),
            # nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 2),
            # nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 4),
            # nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 1, bias=False),
            # 3d and 32 by 32
            # nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.dim_h * 8),  # 40 X 8 = 320
            # nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True),
        )  # ,
        # nn.Conv2d(self.dim_h * 8, 1, 2, 1, 0, bias=False))
        # nn.Conv2d(self.dim_h * 8, 1, 4, 1, 0, bias=False))
        # final layer is fully connected
        self.fc = nn.Linear(self.dim_h * (2 ** 3), self.n_z)

    def forward(self, x):
        # print('enc')
        # print('input ',x.size()) #torch.Size([100, 3,32,32])
        x = self.conv(x)

        x = x.squeeze()
        # print('aft squeeze ',x.size()) #torch.Size([128, 320])
        # aft squeeze  torch.Size([100, 320])
        x = self.fc(x)
        # print('out ',x.size()) #torch.Size([128, 20])
        # out  torch.Size([100, 300])
        return x
