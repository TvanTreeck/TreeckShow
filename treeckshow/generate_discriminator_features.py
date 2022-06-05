import os
import sys
sys.path.append(os.getcwd())

import pandas as pd
from treeckshow.gan import GAN
import os
import torch

def count_dataset_size(feature_dir):
    files = os.listdir(feature_dir)
    pt_files = [os.path.join(feature_dir, f) for f in files if ".pt" in f]
    features = {
        "files": pt_files,
        "labels": ["real" for _ in range(len(pt_files))]
    }
    return len(pt_files), features

def generate(
        n = 1,
        batch_size = 1,
        target_path = "fake_Features",
        sample_mode = "sample",
        model_size="128"
    ):
    os.makedirs(target_path, exist_ok=True)
    gan = GAN(
        config_path=f"models/biggan-deep-{model_size}_config.json",
        model_path=f"models/biggan-deep-{model_size}_pretrained_model.pt"
    )
    n_batches = n // batch_size
    fake_features = {"files": []}
    for bindex in range(n_batches):

        images = gan.inference(batch_size, cls_mode=sample_mode)
        for iindex, image in enumerate(images):
            print("GENERATE DISCRIMINATOR FEATURES:", "batch:", bindex+1, "of", n_batches, "image:", iindex+1, "of", batch_size, end="\r")
            path = os.path.join(target_path, f"image_{bindex}_{iindex}.pt")
            torch.save(image, path)
            fake_features["files"].append(path)

    fake_features["labels"] = ["fake" for _ in range(len(fake_features["files"]))]
    return fake_features


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Prepare data. Precompute Features')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--feats_path', type=str, default="Features", help='path to feature foler')
    parser.add_argument("--target_path", type=str, default="fake_Features", help="path to saved features")
    parser.add_argument("--sample_mode", type=str, default="samples", help="mode of sampling, type class index for train on specific class")
    parser.add_argument("--model_size", type=str, default="128", help="size of the pretrained model")

    args = parser.parse_args()

    n, real_features = count_dataset_size(feature_dir=args.feats_path)
    fake_features = generate(n=n, batch_size=args.batch_size, target_path=args.target_path, sample_mode=args.sample_mode, model_size=args.model_size)
    df = pd.concat(
        [pd.DataFrame(real_features),
        pd.DataFrame(fake_features)]
    )
    df.to_csv(os.path.join(args.target_path, "filelist_feats.txt"), sep="|",header=False, index=False)