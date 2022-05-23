import os
import matplotlib.pyplot as plt
from torchvision.io import read_image
import torch
import pandas as pd

#imgs_path = "Images"
#feats_path = "Features"
#target_shape = (3, 84, 63) # <- HIER EINSTELLEN z.b. (3, 504, 378), (3, 252, 189),(3, 84, 63)

def run(imgs_path="Images", feats_path="Features", target_shape=(3, 84, 63)):
    feats = []
    files = os.listdir(imgs_path)

    if not os.path.isdir(feats_path):
        os.makedirs(feats_path)

    for findex, file in enumerate(files):
        if file == "0.png":
            continue
        filepath = os.path.join(imgs_path, file)
        image = read_image(filepath)

        if target_shape[1]==target_shape[2]:
            min_image_len = min([image.shape[1], image.shape[2]])
            img_sh_1_start = int((image.shape[1] - min_image_len)/2)
            img_sh_2_start = int((image.shape[2] - min_image_len) / 2)
            image = image[:, img_sh_1_start: img_sh_1_start + min_image_len, :]
            image = image[:, :, img_sh_2_start: img_sh_2_start + min_image_len]

        scale_factor = target_shape[1] / image.shape[1]
        image = torch.nn.functional.interpolate(image.unsqueeze(0).float(), scale_factor=scale_factor,
                                                mode="bilinear").int().squeeze()
        if target_shape[0] == 1:
            image = image.float().mean(0).int().unsqueeze(0)

        feat = file.lower().replace(".jpg", ".pt")
        print("PREPARE DATA:","progress:", findex / len(files), image.shape, target_shape, image.shape == target_shape, end="\r")


        if image.shape == target_shape:
            torch.save(image, os.path.join(feats_path, feat))
            feats.append(feat)

        df = pd.DataFrame({"feats": feats})
        df.to_csv(os.path.join(feats_path, "filelist_feats.txt"), index=False, header=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Prepare data. Precompute Features')
    parser.add_argument('--imgs_path', type=str, default="Images", help='path to image foler')
    parser.add_argument('--feats_path', type=str, default="Features", help='path to feature foler')
    parser.add_argument("--img_size", type=int, default=84, help="size of each image dimension")
    parser.add_argument("--img_width", type=int, default=63, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    args = parser.parse_args()

    target_shape = (args.channels, args.img_size, args.img_width)
    run(args.imgs_path, args.feats_path, target_shape)