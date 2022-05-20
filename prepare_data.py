import os
import numpy as np
from torchvision.io import read_image
import torch
import pandas as pd

imgs_path = "Images"
feats_path = "Features"
target_shape=(3, 84, 63) # <- HIER EINSTELLEN z.b. (3, 504, 378), (3, 252, 189),(3, 84, 63)

def run(imgs_path="Images", feats_path="Features", target_shape=(3, 84, 63)):
    feats = []
    files = os.listdir(imgs_path)

    if not os.path.isdir(feats_path):
        os.makedirs(feats_path)

    for findex, file in enumerate(files):
        filepath = os.path.join(imgs_path, file)
        image = read_image(filepath)
        scale_factor = target_shape[1] / image.shape[1]
        image = torch.nn.functional.interpolate(image.unsqueeze(0).float(), scale_factor=scale_factor,
                                                mode="bilinear").int().squeeze()
        feat = file.replace(".jpg", ".pt")
        print("progress:", findex / len(files), image.shape, image.shape == target_shape, )
        torch.save(image, os.path.join(feats_path, feat))
        if image.shape == target_shape:
            feats.append(feat)

        df = pd.DataFrame({"feats": feats})
        df.to_csv(os.path.join(feats_path, "filelist_feats.txt"), index=False, header=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Prepare data. Precompute Features')
    parser.add_argument('--imgs_path', type=str, help='path to image foler')
    parser.add_argument('--feats_path', type=str, help='path to feature foler')
    parser.add_argument("--img_size", type=int, default=84, help="size of each image dimension")
    parser.add_argument("--img_width", type=int, default=63, help="size of each image dimension")
    args = parser.parse_args()

    target_shape = (3, args.img_size, args.img_width)
    run(args.imgs_path, args.feats_path, target_shape)