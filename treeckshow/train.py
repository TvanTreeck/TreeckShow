import os
import sys
sys.path.append(os.getcwd())

import argparse
import numpy as np
import pandas as pd

from torchvision.utils import save_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from treeckshow.gan import GAN
import torch.nn as nn

import torch

os.makedirs("predictions", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.000002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--img_width", type=int, default=128, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=5, help="interval betwen image samples")
parser.add_argument("--ckpt_interval", type=int, default=100, help="interval betwen ckpts")
parser.add_argument("--sample_mode", type=str, default="samples", help="mode of sampling, type class index for train on specific class")
opt = parser.parse_args()

print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_width)

cuda = True if torch.cuda.is_available() else False

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity

gan = GAN(
    config_path=f"models/biggan-deep-{opt.img_size}_config.json",
    model_path=f"models/biggan-deep-{opt.img_size}_pretrained_model.pt"
)

generator = gan.G 

generator.train()
generator.G.train()
# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
#generator = Generator()
discriminator = Discriminator()
state_dict = torch.load("models/discriminator.pt", map_location='cpu' if not torch.cuda.is_available() else None)
discriminator.load_state_dict(state_dict)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
class CustomImageDataset(Dataset):
    def __init__(self, img_filelist, img_dir="/"):
        self.img_dir = img_dir
        self.img_paths = pd.read_csv(img_filelist, header=None)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_paths.iloc[idx, 0])
        # image = read_image(img_path)
        image = torch.load(img_path)
        return image

img_filelist = os.path.join("Features", "filelist_feats.txt")
dataset = CustomImageDataset(img_filelist, "Features")
dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        cls, latent = gan.sample_generator_input(imgs.shape[0], mode=opt.sample_mode)

        # Generate a batch of images
        gen_imgs = generator(z=latent, y=cls)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()
    
        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "TRAIN GAN", "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image((gen_imgs.data[:25] + 1) / 2, "predictions/%d.png" % batches_done, nrow=5, normalize=True)

        if batches_done % opt.ckpt_interval == 0:
            torch.save(discriminator.state_dict(), f"models/discriminator_step2_{batches_done}.pt")

        if batches_done % opt.ckpt_interval == 0:
            torch.save(generator.G.state_dict(), f"models/generator_{batches_done}.pt")