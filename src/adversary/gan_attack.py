import os
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torch.utils.tensorboard import SummaryWriter

from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor

from datetime import datetime


class SquashTransform:
    def __call__(self, inputs):
        return 2 * inputs - 1

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()

        self.linear = nn.Linear(latent_dim, 256 * 4 * 4, False)
        self.lrelu = nn.LeakyReLU(0.2)

        self.convt_1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.convt_2 = nn.ConvTranspose2d(128, 128, 4, 2, 1)
        self.convt_3 = nn.ConvTranspose2d(128, 128, 4, 2, 1)

        self.conv2d_1 = nn.Conv2d(128, 3, 3, 1, 1)

        self.bnorm_2 = nn.BatchNorm2d(128)

        self.tanh = nn.Tanh()

    def forward(self, input):
        input = self.lrelu(self.linear(input)).view(-1, 256, 4, 4)
        input = self.lrelu(self.bnorm_2(self.convt_1(input)))
        input = self.lrelu(self.bnorm_2(self.convt_2(input)))
        input = self.lrelu(self.bnorm_2(self.convt_3(input)))
        input = self.conv2d_1(input)

        return self.tanh(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv2d_1 = nn.Conv2d(3, 32, 4, 2, 1)
        self.conv2d_2 = nn.Conv2d(32, 64, 4, 2, 1)
        self.conv2d_3 = nn.Conv2d(64, 128, 4, 2, 1)
        self.conv2d_4 = nn.Conv2d(128, 1, 4, 1, 0)

        self.lrelu = nn.LeakyReLU(0.2)

        self.norm_1 = nn.BatchNorm2d(64)
        self.norm_2 = nn.BatchNorm2d(128)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        input = self.lrelu(self.conv2d_1(input))
        input = self.lrelu(self.norm_1(self.conv2d_2(input)))
        input = self.lrelu(self.norm_2(self.conv2d_3(input)))
        input = self.conv2d_4(input)

        return self.sigmoid(input)

class VGG19(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = self._make_layers()
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self):
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def define_target():
    model = VGG19()

    if device == 'cuda':
        cudnn.benchmark = True

    model = model.to(DEVICE)

    return model


def define_discriminator():
    model = Discriminator()
    optim = Adam(model.parameters(), lr=LR, betas=(0.5, 0.999))

    return model.to(DEVICE), optim


def define_generator():
    model = Generator(LATENT_DIM)
    optim = Adam(model.parameters(), lr=LR, betas=(0.5, 0.999))

    return model.to(DEVICE), optim


def load_real_samples(n_class, batch_size):
    X = ImageFolder(
        'data/cifar10_attack_labeled_vgg/',
        transform=Compose([
            ToTensor(),
            SquashTransform()
        ])
    )

    idx = np.where(np.array(X.targets) == n_class)[0]
    X_class = Subset(X, idx)

    loader = DataLoader(
        X_class,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )

    return loader


def generate_latent_points(latent_dim, n_samples):
    X = torch.randn(n_samples, latent_dim)
    return X.to(DEVICE)


def generate_fake_samples(g_model, latent_dim, n_samples):
    # generate points in latent space
    inputs = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    generated_images = g_model(inputs)
    # create 'fake' class labels (0)
    zeros = torch.zeros(n_samples, 1).to(DEVICE)
    return generated_images, zeros


def generate_test(g_model):
    generated_images = g_model(X_TEST)

    grid = make_grid(
        generated_images,
        nrow=5,
        padding=5,
        pad_value=1,
        normalize=True
    )

    return grid


def evaluate(model, X, y, total):
    correct = 0

    with torch.no_grad(): # turn off grad
        model.eval() # network in evaluation mode

        outputs = model(X)
        _, predicted = outputs.max(1)
        correct = (predicted == y).sum().item()

    model.train()

    return (100 * correct / total)


def calculate_accuracy(g_model, d_model, dataset):
    # prepare real samples
    X_real, _ = next(iter(dataset))
    X_real = X_real.to(DEVICE)
    y_real = torch.ones(HALF_BATCH, 1).to(DEVICE)

    # evaluate discriminator on real examples
    real_acc = evaluate(d_model, X_real, y_real, HALF_BATCH)

    # prepare fake examples
    X_fake, y_fake = generate_fake_samples(g_model, LATENT_DIM, HALF_BATCH)

    # evaluate discriminator on fake examples
    fake_acc = evaluate(d_model, X_fake, y_fake, HALF_BATCH)

    return real_acc, fake_acc


def trainG(g_model, d_model, g_optim, criterion, alpha=0.5):
    z1 = generate_latent_points(LATENT_DIM, HALF_BATCH)
    z2 = generate_latent_points(LATENT_DIM, HALF_BATCH)

    d_model.eval()

    g_optim.zero_grad()

    images1 = g_model(z1)
    images2 = g_model(z2)

    outputs1 = d_model(images1)
    outputs2 = d_model(images2)

    mode_loss = alpha * torch.mean(torch.abs(images2 - images1)) \
                / torch.mean(torch.abs(z2 - z1))
    image_loss = criterion(outputs1, ONES) + criterion(outputs2, ONES)

    loss = image_loss + mode_loss
    loss.backward()

    g_optim.step()

    d_model.train()

    return loss


def trainD(g_model, d_model, d_optim, criterion, images):
    real_images = images.to(DEVICE)

    g_model.eval()

    fake_images = g_model(
        generate_latent_points(LATENT_DIM, HALF_BATCH)
    )

    real_outputs = d_model(real_images)
    fake_outputs = d_model(fake_images)

    # train on real images
    d_optim.zero_grad()

    real_loss = criterion(real_outputs, ONES)
    real_loss.backward()

    # train on fake images
    d_optim.zero_grad()

    fake_loss = criterion(fake_outputs, ZEROS)
    fake_loss.backward()

    # take a step
    d_optim.step()

    g_model.train()

    return real_loss, fake_loss


def summarize_performance(epoch, sw, d_real_loss, d_fake_loss, g_loss, 
                        real_acc, fake_acc, grid):
    sw.add_scalar(f'GAN_{n_class}/D Real Loss', d_real_loss / STEPS, epoch)
    sw.add_scalar(f'GAN_{n_class}/D Fake Loss', d_fake_loss / STEPS, epoch)

    sw.add_scalar(f'GAN_{n_class}/G Loss', g_loss / STEPS, epoch)

    sw.add_scalar(f'GAN_{n_class}/Real Acc', real_acc, epoch)
    sw.add_scalar(f'GAN_{n_class}/Fake Acc', fake_acc, epoch)

    sw.add_image(f'GAN_{n_class}/Output', grid, epoch)


def train_gan(sw, n_class, dataset, n_epochs):
    d_model, d_optim = define_discriminator()
    g_model, g_optim = define_generator()
    criterion = nn.BCELoss()

    for i in range(n_epochs):
        d_real_loss, d_fake_loss, g_loss = 0, 0, 0

        for j in range(STEPS):
            real_images, _ = next(iter(dataset))
            real_loss, fake_loss = trainD(g_model, d_model, d_optim, criterion, real_images)

            d_real_loss += real_loss
            d_fake_loss += fake_loss

            g_loss += trainG(g_model, d_model, g_optim, criterion)

        if (i) % 10 == 0:
            real_acc, fake_acc = calculate_accuracy(g_model, d_model, dataset)
            grid = generate_test(g_model)

            summarize_performance(
                i, sw,
                d_real_loss, d_fake_loss, g_loss, 
                real_acc, fake_acc,
                grid
            )

    torch.save(g_model, f'models/adversary/g_exp{EXPERIMENT_ID}_class{n_class}.pth')
    torch.save(d_model, f'models/adversary/d_exp{EXPERIMENT_ID}_class{n_class}.pth')


DEVICE = torch.device('cuda' if torch.cuda.is_available else 'cpu')

EPOCHS = 2500

BATCH_SIZE = 32
HALF_BATCH = BATCH_SIZE // 2
STEPS = 500 // BATCH_SIZE

LATENT_DIM = 150

ZEROS = torch.zeros(HALF_BATCH, 1).to(DEVICE)
ONES = torch.ones(HALF_BATCH, 1).to(DEVICE)

LR = 0.0002

X_TEST = generate_latent_points(LATENT_DIM, 10)

EXPERIMENT_ID = len(os.listdir('logs/experiments')) + 1

print(f'#### Experiment {EXPERIMENT_ID} ####')
print(f'Date: {datetime.now().strftime("%Y%m%d_%H-%M")}')

print('\nHiperparametros')
print(f'> Epochs: {EPOCHS}')
print(f'> Learning Rate: {LR}')
print(f'> Batch Size: {BATCH_SIZE}')
print(f'> Half Batch Size: {HALF_BATCH}')
print(f'> Steps: {STEPS}')
print(f'> Latent Dimension: {LATENT_DIM}')
print(f'> Device: {DEVICE}')

print('\nGenerator')
sample_g_model, sample_g_optim = define_generator()
print(sample_g_model)
print(sample_g_optim)

print('\nDiscriminator')
sample_d_model, sample_d_optim = define_generator()
print(sample_d_model)
print(sample_d_optim)

del sample_g_model, sample_d_model, sample_g_optim, sample_d_optim


for n_class in range(10):
    print(f'\n ## Classe {n_class}')

    loader = load_real_samples(n_class, HALF_BATCH)

    print(f'> Dataset: {len(loader.dataset)} samples')

    sw = SummaryWriter(f'logs/experiments/experiment_{EXPERIMENT_ID}')

    train_gan(sw, n_class, loader, EPOCHS)