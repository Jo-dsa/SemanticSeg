import torch
import torch.nn as nn

from src.utils import focal_loss

class Unet(nn.Module):
    def __init__(self, in_channels, out_channels, nb_features, lr=0.0008):
        """
        Implementation of the Unet Architecture

        Args:
            in_channels (integer) : Number of channels from the original image.
            out_channels (integer): Number of channels for the output image.
                                    ``Correspond to the number of class to return``
            nb_features (integer): Number of features to be extracted in the first convolution
            lr (Float): optimizer learning rate ``default: 0.0008``

        """
        super(Unet, self).__init__()

        # attributes
        self.lr=lr
        self.train_loss = []
        self.valid_loss = []
        self.train_accuracy = []
        self.valid_accuracy = []

        # contracting path
        self.conv_1 = self.doubleconv(in_channels, nb_features)
        self.conv_2 = self.doubleconv(nb_features, nb_features*2, maxpool2d=True)
        self.conv_3 = self.doubleconv(nb_features*2, nb_features*4, maxpool2d=True)
        self.conv_4 = self.doubleconv(nb_features*4, nb_features*8, maxpool2d=True)
        
        self.bottleneck = self.doubleconv(nb_features*8, nb_features*16, maxpool2d=True)

        # expansive path
        self.up_4 = self.upsample(nb_features*16, nb_features*8)
        self.deconv_4 = self.doubleconv(nb_features*16, nb_features*8)
        self.up_3 = self.upsample(nb_features*8, nb_features*4)
        self.deconv_3 = self.doubleconv(nb_features*8, nb_features*4)
        self.up_2 = self.upsample(nb_features*4, nb_features*2)
        self.deconv_2 = self.doubleconv(nb_features*4, nb_features*2)
        self.up_1 = self.upsample(nb_features*2, nb_features)
        self.deconv_1 = self.doubleconv(nb_features*2, nb_features)

        self.last_conv = nn.Sequential(
            nn.Conv2d(nb_features, out_channels, kernel_size=1, bias=False),
            nn.Softmax2d()
        )

        # parameters
        self.criterion = focal_loss
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        """
        Defines the forward propagation

        Args:
            x (Tensor): Torch tensor containing the batch of image to process.
                        x of shape BxCxHxW
        """
        # contracting path
        conv_1 = self.conv_1(x)
        conv_2 = self.conv_2(conv_1)
        conv_3 = self.conv_3(conv_2)
        conv_4 = self.conv_4(conv_3)
        # bottleneck
        bottleneck = self.bottleneck(conv_4)
        # expansive path
        deconv_4 = self.deconv_4(torch.cat([self.up_4(bottleneck), conv_4], dim=1))
        deconv_3 = self.deconv_3(torch.cat([self.up_3(deconv_4), conv_3], dim=1))
        deconv_2 = self.deconv_2(torch.cat([self.up_2(deconv_3), conv_2], dim=1))
        deconv_1 = self.deconv_1(torch.cat([self.up_1(deconv_2), conv_1], dim=1))
        
        # last layer
        out = self.last_conv(deconv_1)

        return out

    def doubleconv(self, in_channels, out_channels, maxpool2d=False):
        """
        Defines a double convolution

        Args:
            in_channels (int): Number of channels received
            out_channels (int): Number of output channels
            maxpool2d (boolean): if 'True' add a MaxPool2d layer
        output:
            return a sequential layer
        """
        layers = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        ]

        if maxpool2d:
            layers.insert(0, nn.MaxPool2d(kernel_size=2, stride=2))

        return nn.Sequential(*layers)

    def upsample(self, in_channels, out_channels):
        """
        Defines an umpsampling layer

        Args:
            in_channels (int): Number of channels received
            out_channels (int): Number of output channels
        output:
            return a sequential layer
        """
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        )

        return block
