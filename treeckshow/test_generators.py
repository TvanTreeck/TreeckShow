import os
import sys
sys.path.append(os.getcwd())

from treeckshow.gan import GAN

def init_inference(class_index, model_path, plot_path):
    gan = GAN(
        config_path="models/biggan-deep-128_config.json",
        model_path=model_path
    )
    gan.inference(1, cls_mode=str(class_index), plot=True, plot_path=plot_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test Generators')
    parser.add_argument('--model_path', type=str, default="models", help='path to checkpoints')
    parser.add_argument('--plot_path', type=str, default="generator_samples", help='path to ouput images')
    parser.add_argument('--class_index', type=str, default="1", help='class_index')
    args = parser.parse_args()

    files = os.listdir(args.model_path)
    files = [f for f in files if "generator" in f and ".pt" in f]

    os.makedirs(args.plot_path, exist_ok=True)
    for i, f in enumerate(files):
        print(f, "progress",  i/len(files))
        model_path = os.path.join(args.model_path, f)
        image_path = os.path.join(args.plot_path, f.replace(".pt", ".png"))
        init_inference(class_index=args.class_index, model_path=args.model_path, plot_path=args.plot_path)
