import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, submodule=None, innermost=False, outermost=False):
        super(UNetBlock, self).__init__()
        self.outermost = outermost
        
        downconv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(out_channels)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(in_channels)

        if outermost:
            upconv = nn.ConvTranspose2d(out_channels * 2, in_channels, kernel_size=4, stride=2, padding=1)
            model = [downconv] + [submodule] + [uprelu, upconv, nn.Tanh()]
        elif innermost:
            upconv = nn.ConvTranspose2d(out_channels, in_channels, kernel_size=4, stride=2, padding=1, bias=False)
            model = [downrelu, downconv, uprelu, upconv, upnorm]
        else:
            upconv = nn.ConvTranspose2d(out_channels * 2, in_channels, kernel_size=4, stride=2, padding=1, bias=False)
            model = [downrelu, downconv, downnorm, submodule, uprelu, upconv, upnorm]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            # Skip Connection: Concatenate along the channel dimension
            return torch.cat([x, self.model(x)], 1)

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_filters=64):
        super(UNetGenerator, self).__init__()
        # Build U-Net from the inside out
        block = UNetBlock(num_filters * 8, num_filters * 8, submodule=None, innermost=True)
        for _ in range(3):
            block = UNetBlock(num_filters * 8, num_filters * 8, submodule=block)
        block = UNetBlock(num_filters * 4, num_filters * 8, submodule=block)
        block = UNetBlock(num_filters * 2, num_filters * 4, submodule=block)
        block = UNetBlock(num_filters, num_filters * 2, submodule=block)
        self.model = UNetBlock(out_channels, num_filters, submodule=block, outermost=True)

    def forward(self, x):
        return self.model(x)