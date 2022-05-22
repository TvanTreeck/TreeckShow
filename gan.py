from pytorch_pretrained_gans import BigGAN
import torch
import matplotlib.pyplot as plt

class GAN:
    def __init__(self, config_path="models/config.json", model_path="models/pretrained_model.pth.tar"):
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

    def inference(self, plot=False, plot_path="plot.png"):
        cls, latent = self. sample_generator_input(1)
        image = self.G(z=latent, y=cls)
        image = image.squeeze(0).transpose(1,0).transpose(2,1).detach().numpy()

        if plot:
            plt.figure()
            plt.imshow(image)
            plt.savefig(plot_path)

if __name__ == "__main__":
    gan = GAN(config_path="models/config.json", model_path="models/pretrained_model.pth.tar")
    gan.inference(plot=True)
