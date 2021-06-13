import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
#from torchvision.transforms.transforms import Normalize

from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Normalize, ToTensor, Resize

from datetime import datetime

from fid import read_stats_file
from utils import SquashTransform
from utils import encode_onehot
from utils import generate_conditional_latent_points
from experiment import ExperimentLog, generate_experiment_id

from models.dcgan_32 import Generator
from models.dcgan_32 import Discriminator
from models.vgg import VGG16


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def define_discriminator():
    model = Discriminator(n_class=N_CLASS).to(DEVICE)
    model.apply(weights_init)

    optim = Adam(model.parameters(), lr=LR, amsgrad=True)
    lrdecay = ExponentialLR(optimizer=optim, gamma=D_LR_DECAY)

    return model, optim, lrdecay


def define_generator():
    model = Generator(LATENT_DIM, n_class=N_CLASS).to(DEVICE)
    model.apply(weights_init)

    optim = Adam(model.parameters(), lr=LR, amsgrad=True)
    lrdecay = ExponentialLR(optimizer=optim, gamma=G_LR_DECAY)

    return model, optim, lrdecay


def load_real_samples():
    X = ImageFolder(
        root=DATAROOT,
        transform=Compose([
            Resize(IMAGE_SIZE),
            ToTensor(),
            SquashTransform(),
        ])
    )

    loader = DataLoader(
        X,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKERS,
        drop_last=True
    )

    del X

    return loader


def generate_test(g_model):
    generated_images = None 

    with torch.no_grad():
        generated_images = g_model(TEST_FIXED_NOISE, TEST_FIXED_CONDITIONS)

    grid = make_grid(
        generated_images,
        nrow=5,
        padding=5,
        pad_value=1,
        normalize=True
    )

    return grid


def summarize_performance(epoch, sw, d_loss, g_loss, g_mode_loss, d_x, d_g_z, g_lr, d_lr, grid):
    sw.add_scalar(f'DCGAN/D Loss', d_loss, epoch)
    sw.add_scalar(f'DCGAN/G Loss', g_loss, epoch)
    sw.add_scalar(f'DCGAN/G Mode Loss', g_mode_loss, epoch)

    sw.add_scalar(f'DCGAN/D(x)', d_x, epoch)
    sw.add_scalar(f'DCGAN/D(G(z))', d_g_z, epoch)

    sw.add_scalar(f'DCGAN/G LR', g_lr, epoch)
    sw.add_scalar(f'DCGAN/D LR', d_lr, epoch)

    sw.add_image(f'DCGAN/Output', grid, epoch)


def calculate_fid(images, batch_size, sw, epoch):
    # load model
    from fid import FID
    fid_model = FID()

    # calculate statistics
    fake_mu, fake_sigma = fid_model.calculate_statistics(images, batch_size)

    # calculate FID
    fid_odd = fid_model.calculate_fid(fake_mu, fake_sigma, TARGET_FID[0], TARGET_FID[1])
    fid_real = fid_model.calculate_fid(fake_mu, fake_sigma, DATASET_FID[0], DATASET_FID[1])

    # save FID
    sw.add_scalar(f'DCGAN/FID ODD', fid_odd, epoch)
    sw.add_scalar(f'DCGAN/FID Real', fid_real, epoch)

    # clean memory
    del fid_model, images, fake_mu, fake_sigma

    return fid_real


def calculate_target_acc(images, conditions, sw, epoch):
    # Load target model
    target = VGG16(out_features=N_CLASS)
    target = target.to(DEVICE)
    checkpoint = torch.load('models/target/mnist.vgg16.pth')

    target = nn.DataParallel(target)
    cudnn.benchmark = True
    target.load_state_dict(checkpoint['net'])

    del checkpoint

    _, labels = conditions.max(1)
    # labels = labels.to(DEVICE)

    SIZE = 32
    # norm = Normalize((0.43767047, 0.44375867, 0.47279018), (0.19798356, 0.20096427, 0.19697163))
    norm = Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))

    images = F.interpolate(images, size=SIZE)

    for i in range(images.shape[0]):
        images[i] = norm(images[i])
    
    with torch.no_grad(): # turn off grad
        target.eval()

        outputs = target(images)
        _, predicted = outputs.max(1)
        correct = predicted.eq(labels).sum().item()

    acc = 100 * correct / images.size(0)

    sw.add_scalar(f'DCGAN/Target Acc', acc, epoch)

    return acc

def categorical_criterion(input, target):
    _, labels = target.max(dim=1)

    return nn.CrossEntropyLoss()(input, labels)

def train_gan(sw, dataloader):
    d_model, d_optim, d_lrdecay = define_discriminator()
    g_model, g_optim, g_lrdecay = define_generator()
    binary_criterion = nn.BCELoss()

    G_losses = []
    D_losses = []

    # global best_fid
    # best_fid = np.inf

    for epoch in range(EPOCHS):
        D_x, D_G_z1, D_G_z2 = [], [], []
        errD, errG = torch.tensor(0), torch.tensor(0)
        mode_loss = 0

        for _, data in enumerate(dataloader):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            d_model.zero_grad()
            # Format batch
            real_images = data[0].to(DEVICE)
            conditions = encode_onehot(data[1], N_CLASS).to(DEVICE)
            mini_batch_size = real_images.size(0)
            label = torch.full((mini_batch_size,), REAL_LABEL, dtype=torch.float, device=DEVICE)
            # Forward pass real batch through D
            output_valid, output_clf = d_model(real_images)
            # Calculate loss on all-real batch
            l_valid = binary_criterion(output_valid.view(-1), label)
            l_clf = categorical_criterion(output_clf, conditions) 
            errD_real = l_valid + l_clf
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x.append(output_valid.mean().item())

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise, conditions = generate_conditional_latent_points(LATENT_DIM, mini_batch_size, N_CLASS, DEVICE)
            # Generate fake image batch with G
            fake = g_model(noise, conditions)
            label.fill_(FAKE_LABEL)
            # Classify all fake batch with D
            output_valid, output_clf = d_model(fake.detach())
            # Calculate D's loss on the all-fake batch
            l_valid = binary_criterion(output_valid.view(-1), label)
            l_clf = categorical_criterion(output_clf, conditions) 
            errD_fake = l_valid + l_clf
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1.append(output_valid.mean().item())
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            d_optim.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            g_model.zero_grad()
            # Format batch
            mini_batch_size = mini_batch_size * 2
            label = torch.full((mini_batch_size,), REAL_LABEL, dtype=torch.float, device=DEVICE)
            # Generate batch of latent vectors
            noise, conditions = generate_conditional_latent_points(LATENT_DIM, mini_batch_size, N_CLASS, DEVICE)
            # Generate fake image batch with G
            fake = g_model(noise, conditions) 
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output_valid, output_clf = d_model(fake)
            # Calculate mode seeking loss
            z1, z2, *_ = torch.split(noise, noise.size(0)//2)
            f1, f2, *_ = torch.split(fake, fake.size(0)//2)
            mode_loss = torch.mean(torch.abs(f2 - f1)) / torch.mean(torch.abs(z2 - z1))
            mode_loss = LAMBDA_MODE_LOSS * mode_loss
            mode_loss = 1 / (mode_loss + 1e-5)
            # Calculate G's loss based on this output            
            l_valid = binary_criterion(output_valid.view(-1), label)
            l_clf = categorical_criterion(output_clf, conditions) 
            errG = l_valid + l_clf 
            loss_G = errG + mode_loss
            # Calculate gradients for G
            loss_G.backward()
            D_G_z2.append(output_valid.mean().item())
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
        if ((epoch + 1) % 100 == 0) or (epoch == 0):
            summarize_performance(
                epoch, sw,
                errD.item(), errG.item(), mode_loss,
                D_x, D_G_z1,
                g_lrdecay.get_last_lr()[-1], d_lrdecay.get_last_lr()[-1],
                # LR, LR,
                generate_test(g_model)
            )

            # generate fake samples
            fake_images = g_model(FID_FIXED_NOISE, FID_FIXED_CONDITIONS)

            # saving space in the gpu for loading inception v3
            g_model = g_model.to(torch.device('cpu'))
            d_model = d_model.to(torch.device('cpu'))

            #if epoch % 50 == 0:
            calculate_fid(fake_images, BATCH_SIZE, sw, epoch)
            calculate_target_acc(fake_images, FID_FIXED_CONDITIONS, sw, epoch)

            # put models back on GPU
            g_model = g_model.to(DEVICE)
            d_model = d_model.to(DEVICE)

    torch.save(g_model, f'models/gan/g_exp{EXPERIMENT_ID}.pth')
    torch.save(d_model, f'models/gan/d_exp{EXPERIMENT_ID}.pth')


# Generate the next experiment ID
EXPERIMENT_ID = generate_experiment_id()

# Create the experiment folder
EXPERIMENT_PATH = f'logs/mnist/single_gan/experiment_{EXPERIMENT_ID}'

if not os.path.exists(EXPERIMENT_PATH):
    os.mkdir(EXPERIMENT_PATH)

# Get the experiment log
LOGGER = ExperimentLog(f"{EXPERIMENT_PATH}/log_gan")

# Set random seed for reproducibility
manualSeed = 999 
# manualSeed = random.randint(1, 10000) # use if you want new results
random.seed(manualSeed)
torch.manual_seed(manualSeed)
LOGGER.write(f"Random Seed: {manualSeed}")

# Get best available device
DEVICE = torch.device('cuda' if torch.cuda.is_available else 'cpu')

# Parameters
DATAROOT = 'data/mnist.vgg16/dataset_iv_sl/'
WORKERS = 4
N_CLASS = 10 
BATCH_SIZE = 16
IMAGE_SIZE = 32
LATENT_DIM = 100
LAMBDA_MODE_LOSS = 2 
LR = 0.0005
EPOCHS = 500
D_LR_DECAY = 0.99
G_LR_DECAY = 0.99
TARGET_FID = read_stats_file('logs/fid/mnist_od_fid.npz')
DATASET_FID = read_stats_file('logs/fid/mnist_npd_sl_fid.npz')

TEST_FIXED_NOISE, TEST_FIXED_CONDITIONS = generate_conditional_latent_points(LATENT_DIM, 20, N_CLASS, DEVICE)

FID_FIXED_NOISE, FID_FIXED_CONDITIONS = generate_conditional_latent_points(LATENT_DIM, 100, N_CLASS, DEVICE)

REAL_LABEL = 1
FAKE_LABEL = 0

LOGGER.write(f'#### Experiment {EXPERIMENT_ID} ####')
LOGGER.write(f'Date: {datetime.now().strftime("%Y%m%d_%H-%M")}')

LOGGER.write('\nHiperparametros')
LOGGER.write(f'> Epochs: {EPOCHS}')
LOGGER.write(f'> Learning Rate: {LR}')
LOGGER.write(f'> Image Size: {IMAGE_SIZE}')
LOGGER.write(f'> Image Size: {IMAGE_SIZE}')
LOGGER.write(f'> Batch Size: {BATCH_SIZE}')
LOGGER.write(f'> Lambda Mode Loss: {LAMBDA_MODE_LOSS}')
LOGGER.write(f'> Latent Dimension: {LATENT_DIM}')
LOGGER.write(f'> Device: {DEVICE}')

LOGGER.write('\nGenerator')
sample_g_model, sample_g_optim, sample_g_lrdecay = define_generator()
# sample_g_model, sample_g_optim = define_generator()
LOGGER.write(sample_g_model)
LOGGER.write(sample_g_optim)
LOGGER.write(sample_g_lrdecay.__class__.__name__)
LOGGER.write(f'Gamma: {G_LR_DECAY}')

LOGGER.write('\nDiscriminator')
sample_d_model, sample_d_optim, sample_d_lrdecay = define_discriminator()
# sample_d_model, sample_d_optim = define_discriminator()
LOGGER.write(sample_d_model)
LOGGER.write(sample_d_optim)
LOGGER.write(sample_d_lrdecay.__class__.__name__)
LOGGER.write(f'Gamma: {D_LR_DECAY}')

del sample_g_model, sample_d_model, sample_g_optim, sample_d_optim #, sample_g_lrdecay, sample_d_lrdecay

LOGGER.write("Starting GAN attack")

loader = load_real_samples()
LOGGER.write(f'> Dataset: {len(loader.dataset)} samples')

sw = SummaryWriter(EXPERIMENT_PATH)

train_gan(sw, loader)
