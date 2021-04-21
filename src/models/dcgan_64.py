import torch
import torch.nn as nn

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of feature maps in generator
ngf = 64
ndf = 64

class Generator(nn.Module):
    def __init__(self, latent_dim, n_class):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(
                in_channels=latent_dim + n_class, 
                out_channels=ngf*8,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(num_features=ngf*8),
            nn.ReLU(inplace=True),

            # state size => (ngf*8) x 4 x 4
            nn.ConvTranspose2d(
                in_channels=ngf*8, 
                out_channels=ngf*4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=ngf*4),
            nn.ReLU(inplace=True),

            # state size => (ngf*4) x 8 x 8
            nn.ConvTranspose2d(
                in_channels=ngf*4, 
                out_channels=ngf*2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=ngf*2),
            nn.ReLU(inplace=True),

            # state size => (ngf*2) x 16 x 16
            nn.ConvTranspose2d(
                in_channels=ngf*2, 
                out_channels=ngf,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=ngf),
            nn.ReLU(inplace=True),

            # state size => (ngf) x 32 x 32
            nn.ConvTranspose2d(
                in_channels=ngf, 
                out_channels=nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.Tanh(),

            # state size => (nc) x 64 x 64 
        )

    def forward(self, inputs, condition):
        # concatenate noise and condition
        concatenated_input = torch.cat(
            (inputs, condition),
            dim=1
        )

        # reshape the latent vector into a feature map
        concatenated_input = concatenated_input.unsqueeze(2).unsqueeze(3)

        return self.main(concatenated_input)

class Discriminator(nn.Module):
    def __init__(self, n_class):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # state size => (ndf) x 64 x 64 
            nn.Conv2d(
                in_channels=nc, 
                out_channels=ndf,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True
            ),

            # state size => (ndf) x 32 x 32 
            nn.Conv2d(
                in_channels=ndf, 
                out_channels=ndf*2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=ndf*2),
            nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True
            ),

            # state size => (ndf*2) x 16 x 16 
            nn.Conv2d(
                in_channels=ndf*2, 
                out_channels=ndf*4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=ndf*4),
            nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True
            ),

            # state size => (ndf*4) x 8 x 8 
            nn.Conv2d(
                in_channels=ndf*4, 
                out_channels=ndf*8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=ndf*8),
            nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True
            ),
        )

        # categorical classifier
        self.clf = nn.Sequential(
            nn.Linear(
                in_features=ndf*8*4*4,
                out_features=n_class,
                bias=True
            ),
            nn.Softmax(dim=1)
        )
        
        # real / fake classifier
        self.police = nn.Sequential(
            # state size => (ndf*8) x 4 x 4 
            nn.Conv2d(
                in_channels=ndf*8,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        features = self.main(inputs)
        is_true = self.police(features).view(-1, 1)
        class_predicted = self.clf(features.view(features.shape[0], -1))

        return is_true, class_predicted
