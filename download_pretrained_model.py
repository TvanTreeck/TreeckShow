import torch
from pytorch_pretrained_gans.BigGAN import make_biggan
import matplotlib.pyplot as plt

# Sample a class-conditional image from BigGAN with default resolution 256
gan_type='biggan-deep-128'
G = make_biggan(model_name=gan_type)

state_dict  = G.G.state_dict()
config = G.G.config

torch.save(state_dict, "models/"+gan_type+"_pretrained_model.pth.tar")
             
with open(gan_type+"_config.json", "w") as f:
    f.write(config.to_json_string())
             

for i in range(5):
    y = G.sample_class(batch_size=1)  # -> torch.Size([1, 1000])
    z = G.sample_latent(batch_size=1)  # -> torch.Size([1, 128])
    x = G(z=z, y=y)  

    plt.figure()
    plt.imshow(x.squeeze(0).transpose(1,0).transpose(2,1).detach().numpy())
    plt.savefig("models/"+gan_type+f"pretrained_sample_{i}")
