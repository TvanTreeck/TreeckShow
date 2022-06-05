import os
import sys
sys.path.append(os.getcwd())

from pytorch_pretrained_gans import BigGAN
import torch
import matplotlib.pyplot as plt
import os

class GAN:
    def __init__(
            self,
            config_path="models/biggan-deep-128_config.json",
            model_path="models/biggan-deep-128_pretrained_model.pt"
    ):
        self.config = BigGAN.config.BigGANConfig()
        self.config = self.config.from_json_file(config_path)

        state_dict = torch.load(model_path, map_location='cpu' if not torch.cuda.is_available() else None)

        self.G = BigGAN.model.BigGAN(self.config).eval()
        self.G.load_state_dict(state_dict, strict=False)
        self.G = BigGAN.GeneratorWrapper(self.G).eval()

    def sample_generator_input(self, n, mode="sample"):
        cls = self.G.sample_class(batch_size=n)

        if mode == "mixed":
            cls[:, :] = 1/1000
        if mode.isnumeric():
            cls = torch.zeros([1, 1000])
            cls[:, int(mode)] = 1.0

        latent = self.G.sample_latent(batch_size=n)
        return cls, latent

    def inference(self, n=1, plot=False, plot_path="", cls_mode="sample", prefix="plot"):
        cls, latent = self.sample_generator_input(n, cls_mode)
        images = self.G(z=latent, y=cls)
        for index, image in enumerate(images):
            image = (image.transpose(1,0).transpose(2,1).detach().numpy() + 1) / 2
            if plot:
                plt.figure()
                plt.imshow(image)
                if ".png" in plot_path or ".jpg" in plot_path:
                    ppath = plot_path
                else:
                    ppath = os.path.join(plot_path, f"{prefix}_{index}.png")
                plt.savefig(ppath)
        return images

    def plot_classes(self):
        os.makedirs("classes", exist_ok=True)
        for i in range(1000):
            print("PLOT CLASSES:", "progress:", i / 1000, "class:", i, end="\r")

            self.inference(1, True, "classes", str(i), f"class_{i}")


if __name__ == "__main__":
    gan = GAN(
        config_path="models/biggan-deep-256_config.json",
        model_path="models/biggan-deep-256_pretrained_model.pt"
    )
    gan.inference(1, plot=True)
    gan.plot_classes()
