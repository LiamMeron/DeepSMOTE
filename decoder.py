from torch import nn as nn


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.n_channel = args["n_channel"]
        self.dim_h = args["dim_h"]
        self.n_z = args["n_z"]

        # first layer is fully connected
        self.fc = nn.Sequential(nn.Linear(self.n_z, self.dim_h * 8 * 7 * 7), nn.ReLU())

        # deconvolutional filters, essentially inverse of convolutional filters
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.dim_h * 8, self.dim_h * 4, 4),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, 4),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 2, 1, 4, stride=2),
            # nn.Sigmoid())
            nn.Tanh(),
        )

    def forward(self, x):
        # print('dec')
        # print('input ',x.size())
        x = self.fc(x)
        x = x.view(-1, self.dim_h * 8, 7, 7)
        x = self.deconv(x)
        return x
