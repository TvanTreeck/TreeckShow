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

    def sample_generator_input(self, n):
        cls = self.G.sample_class(batch_size=n)  # -> torch.Size([1, 1000])
        latent = self.G.sample_latent(batch_size=n)
        return cls, latent

    def inference(self, n=1, plot=False, plot_path=""):
        cls, latent = self. sample_generator_input(n)
        images = self.G(z=latent, y=cls)
        for index, image in enumerate(images):
            image = image.transpose(1,0).transpose(2,1).detach().numpy()
            if plot:
                plt.figure()
                plt.imshow(image)
                plt.savefig(os.path.join(plot_path, f"plot_{index}.png"))
        return images

if __name__ == "__main__":
    gan = GAN(
        config_path="models/biggan-deep-128_config.json",
        model_path="models/biggan-deep-128_pretrained_model.pth.tar"
    )
    gan.inference(1, plot=True)
