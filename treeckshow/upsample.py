import os
import sys
sys.path.append(os.getcwd())

import torch
from torchvision.io import read_image
import matplotlib.pyplot as plt

def upsample(filepath, plot_path, scale_factor):
    image = read_image(filepath)

    image = torch.nn.functional.interpolate(
        image.unsqueeze(0).float(),
        scale_factor=scale_factor,
        mode="bilinear"
    ).int().squeeze()

    image = image.transpose(1, 0).transpose(2, 1).detach().numpy()

    plt.figure()
    plt.imshow(image)
    plt.savefig(os.path.join(plot_path, filepath.split("/")[-1]))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Upsample image')
    parser.add_argument('--filepath', type=str, default="predictions/0.png", help='path to input image')
    parser.add_argument('--plot_path', type=str, default="upsample_images", help='path to ouput image')
    parser.add_argument('--scale_factor', type=float, default=5.0, help='scale_factor')
    args = parser.parse_args()

    os.makedirs(args.plot_path, exist_ok=True)
    upsample(args.filepath, args.plot_path, args.scale_factor)