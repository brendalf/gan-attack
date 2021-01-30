import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR

from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, CenterCrop, Normalize, Resize

from datetime import datetime

from utils import SquashTransform, generate_latent_points
from experiment import ExperimentLog, generate_experiment_id
from models.generator import GeneratorDCGAN as Generator
from models.discriminator import DiscriminatorDCGAN as Discriminator


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def define_discriminator():
    model = Discriminator().to(DEVICE)
    model.apply(weights_init)

    optim = Adam(model.parameters(), lr=LR*0.5, amsgrad=True)
    lrdecay = ExponentialLR(optimizer=optim, gamma=0.999)
    #lrdecay = MultiStepLR(optimizer=optim, milestones=np.arange(100, EPOCHS, 100), gamma=0.99)

    return model, optim, lrdecay


def define_generator():
    model = Generator(LATENT_DIM).to(DEVICE)
    model.apply(weights_init)

    optim = Adam(model.parameters(), lr=LR, amsgrad=True)
    lrdecay = ExponentialLR(optimizer=optim, gamma=0.999)
    #lrdecay = MultiStepLR(optimizer=optim, milestones=np.arange(100, EPOCHS, 100), gamma=0.99)

    return model, optim, lrdecay


def load_real_samples(n_class):
    X = ImageFolder(
        root=DATAROOT,
        transform=Compose([
            Resize(IMAGE_SIZE),
            CenterCrop(IMAGE_SIZE),
            ToTensor(),
            SquashTransform(),
        ])
    )

    idx = np.where(np.array(X.targets) == n_class)[0]
    X_class = Subset(X, idx)

    loader = DataLoader(
        X_class,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKERS,
    )

    return loader


def generate_test(g_model):
    with torch.no_grad():
        generated_images = g_model(FIXED_NOISE)

    grid = make_grid(
        generated_images,
        nrow=5,
        padding=5,
        pad_value=1,
        normalize=True
    )

    return grid


def summarize_performance(epoch, sw, n_class, d_loss, g_loss, d_x, d_g_z, g_lr, d_lr, grid):
    sw.add_scalar(f'GAN {n_class}/D Loss', d_loss, epoch)
    sw.add_scalar(f'GAN {n_class}/G Loss', g_loss, epoch)

    sw.add_scalar(f'GAN {n_class}/D(x)', d_x, epoch)
    sw.add_scalar(f'GAN {n_class}/D(G(z))', d_g_z, epoch)

    sw.add_scalar(f'GAN {n_class}/G LR', g_lr, epoch)
    sw.add_scalar(f'GAN {n_class}/D LR', d_lr, epoch)

    sw.add_image(f'GAN {n_class}/Output', grid, epoch)


def calculate_fid(g_model, d_model, loader, sw, epoch, n_class):
    # generate fake samples
    fake_images = g_model(generate_latent_points(LATENT_DIM, 32, DEVICE))
    #fake_images = g_model(FIXED_NOISE)

    # load real samples
    b1 = next(iter(loader))
    b2 = next(iter(loader))
    real_images = torch.cat((b1[0], b2[0]), 0)
    real_images.to(DEVICE)
    del b1, b2

    # saving space in the gpu for loading inception v3
    g_model = g_model.to(torch.device('cpu'))
    d_model = d_model.to(torch.device('cpu'))

    # load model
    from fid_pytorch import FID
    fid_model = FID()

    # calculate FID
    fid = fid_model.calculate_fid(real_images, fake_images)

    # save FID
    sw.add_scalar(f'GAN {n_class}/FID', fid, epoch)

    # clean memory
    del fid_model, real_images, fake_images

    # put models back on GPU
    g_model = g_model.to(DEVICE)
    d_model = d_model.to(DEVICE)

    return fid


def train_gan(sw, n_class, dataloader):
    d_model, d_optim, d_lrdecay = define_discriminator()
    g_model, g_optim, g_lrdecay = define_generator()
    criterion = nn.BCELoss()

    img_list = []
    G_losses = []
    D_losses = []

    min_fid = np.inf

    for epoch in range(EPOCHS):
        D_x, D_G_z1, D_G_z2 = [], [], []

        for i, data in enumerate(dataloader):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            d_model.zero_grad()
            # Format batch
            real_images = data[0].to(DEVICE)
            mini_batch_size = real_images.size(0)
            label = torch.full((mini_batch_size,), REAL_LABEL, dtype=torch.float, device=DEVICE)
            # Forward pass real batch through D
            output = d_model(real_images).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x.append(output.mean().item())

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = generate_latent_points(LATENT_DIM, mini_batch_size, DEVICE)
            # Generate fake image batch with G
            fake = g_model(noise)
            label.fill_(FAKE_LABEL)
            # Classify all fake batch with D
            output = d_model(fake.detach()).view(-1)
            # Calculate mode loss
            #mode_loss = alpha * torch.mean(torch.abs(images2 - images1)) \
            #    / torch.mean(torch.abs(z2 - z1))
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1.append(output.mean().item())
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            d_optim.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            g_model.zero_grad()
            label.fill_(REAL_LABEL)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = d_model(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2.append(output.mean().item())
            # Update G
            g_optim.step()

        # Learning rate decay
        d_lrdecay.step()
        g_lrdecay.step()

        # Calculate the mean
        D_x = np.mean(D_x)
        D_G_z1 = np.mean(D_G_z1)
        D_G_z2 = np.mean(D_G_z2)

        # Output training stats
        LOGGER.write('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
            % (epoch+1, EPOCHS, errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        fid = min_fid
        if epoch % 50 == 0:
            summarize_performance(
                epoch, sw, n_class,
                errD.item(), errG.item(), 
                D_x, D_G_z1,
                g_lrdecay.get_last_lr()[-1], d_lrdecay.get_last_lr()[-1],
                generate_test(g_model)
            )

            #if epoch % 50 == 0:
            fid = calculate_fid(g_model, d_model, dataloader, sw, epoch, n_class)

        if fid < min_fid:
            min_fid = fid

            torch.save(g_model, f'models/adversary/g_exp{EXPERIMENT_ID}_class{n_class}.pth')
            torch.save(d_model, f'models/adversary/d_exp{EXPERIMENT_ID}_class{n_class}.pth')


# Generate the next experiment ID
EXPERIMENT_ID = generate_experiment_id()

# Create the experiment folder
EXPERIMENT_PATH = f'logs/experiments/experiment_{EXPERIMENT_ID}'

if not os.path.exists(EXPERIMENT_PATH):
    os.mkdir(EXPERIMENT_PATH)

# Get the experiment log
LOGGER = ExperimentLog(f"{EXPERIMENT_PATH}/log")

# Set random seed for reproducibility
manualSeed = 999 
# manualSeed = random.randint(1, 10000) # use if you want new results
random.seed(manualSeed)
torch.manual_seed(manualSeed)
LOGGER.write(f"Random Seed: {manualSeed}")

# Get best available device
DEVICE = torch.device('cuda' if torch.cuda.is_available else 'cpu')

# Parameters
DATAROOT = 'data/cifar10_attack_labeled_vgg/'
WORKERS = 4
BATCH_SIZE = 16
IMAGE_SIZE = 64
LATENT_DIM = 100
LR = 0.0005
EPOCHS = 1500

FIXED_NOISE = generate_latent_points(LATENT_DIM, 20, DEVICE)
REAL_LABEL = 1
FAKE_LABEL = 0

LOGGER.write(f'#### Experiment {EXPERIMENT_ID} ####')
LOGGER.write(f'Date: {datetime.now().strftime("%Y%m%d_%H-%M")}')

LOGGER.write('\nHiperparametros')
LOGGER.write(f'> Epochs: {EPOCHS}')
LOGGER.write(f'> Learning Rate: {LR}')
LOGGER.write(f'> Image Size: {IMAGE_SIZE}')
LOGGER.write(f'> Batch Size: {BATCH_SIZE}')
LOGGER.write(f'> Latent Dimension: {LATENT_DIM}')
LOGGER.write(f'> Device: {DEVICE}')

LOGGER.write('\nGenerator')
sample_g_model, sample_g_optim, sample_g_lrdecay = define_generator()
LOGGER.write(sample_g_model)
LOGGER.write(sample_g_optim)
LOGGER.write(sample_g_lrdecay.__class__.__name__)

LOGGER.write('\nDiscriminator')
sample_d_model, sample_d_optim, sample_d_lrdecay = define_discriminator()
LOGGER.write(sample_d_model)
LOGGER.write(sample_d_optim)
LOGGER.write(sample_d_lrdecay.__class__.__name__)

del sample_g_model, sample_d_model, sample_g_optim, sample_d_optim, sample_g_lrdecay, sample_d_lrdecay

LOGGER.write("Starting GAN attack")
for n_class in range(10):
    LOGGER.write(f'\n ## Classe {n_class}')

    loader = load_real_samples(n_class)

    LOGGER.write(f'> Dataset: {len(loader.dataset)} samples')

    sw = SummaryWriter(EXPERIMENT_PATH)

    train_gan(sw, n_class, loader)