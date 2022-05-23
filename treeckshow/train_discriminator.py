import argparse
import os
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn

import torch

os.makedirs("predictions", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--img_width", type=int, default=128, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--ckpt_interval", type=int, default=500, help="interval betwen ckpts")
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


# Loss function
adversarial_loss = torch.nn.BCELoss() # TODO

# Initialize discriminator
discriminator = Discriminator()
if os.path.isfile("models/best_discriminator.pt"):
    state_dict = torch.load("models/best_discriminator.pt",
                            map_location='cpu' if not torch.cuda.is_available() else None)
    discriminator.load_state_dict(state_dict)

if cuda:
    discriminator.cuda()
    adversarial_loss.cuda()


# Configure data loader
class CustomImageLabelDataset(Dataset):
    def __init__(self, img_filelist, img_dir=""):
        self.img_dir = img_dir
        self.img_paths = pd.read_csv(img_filelist, header=None, sep="|")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_paths.iloc[idx, 0])
        image = torch.load(img_path).float()

        label = self.img_paths.iloc[idx, 1]
        label_tensor = Tensor(1)
        if label == "real":
            label_tensor = label_tensor.fill_(1.0)
        elif label == "fake":
            label_tensor = label_tensor.fill_(0.0)
        label_tensor = torch.autograd.Variable(label_tensor)
        return image, label_tensor

def accuracy(prediction, labels):
    pred = (prediction>0.5).int()
    tp, tn, fp, fn = 0,0,0,0
    for i in range(len(labels)):
        l = labels[i].int().item()
        ap = pred[i].item()
        if l == 1:
            if ap == 1:
                tp += 1
            elif ap == 0:
                tn += 1
        elif l == 0:
            if ap == 1:
                fp += 1
            elif ap == 0:
                fn += 1

    return (pred==labels).sum()/len(labels), (tp, tn, fp, fn)

img_filelist = os.path.join("fake_Features", "filelist_feats.txt")
dataset = CustomImageLabelDataset(img_filelist)
dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

# Optimizers
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    cmat = {
        "tp": 0, "tn": 0, "fp": 0, "fn": 0,
    }
    for i, (imgs, labels) in enumerate(dataloader):
        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        prediction = discriminator(imgs)
        d_loss = adversarial_loss(prediction, labels)
        batch_accuracy, (tp, tn, fp, fn) = accuracy(prediction, labels)
        cmat["tp"]+=tp
        cmat["tn"] += tn
        cmat["fp"] += fp
        cmat["fn"] += fn
        d_loss.backward()
        optimizer_D.step()

        print(
            "TRAIN DISCRIMINATOR:",
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [D acc: %f] [D mean_pred: %f] [cmat tp: %f tn: %f fp: %f fn: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), batch_accuracy.item(), prediction.mean().item(), cmat["tp"], cmat["tn"], cmat["fp"], cmat["fn"])
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.ckpt_interval == 0:
            torch.save(discriminator.state_dict(), f"models/discriminator_{batches_done}.pt")

torch.save(discriminator.state_dict(), f"models/discriminator.pt")