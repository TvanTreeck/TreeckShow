import os
import sys
sys.path.append(os.getcwd())

import torch
from pytorch_pretrained_gans.BigGAN import make_biggan

# Sample a class-conditional image from BigGAN with default resolution 256
gan_type='biggan-deep-128'

if not os.path.isdir("models"):
    os.makedirs("models")

if not os.path.isfile("models/"+gan_type+"_pretrained_model.pt"):
    print("DOWNLOAD MODELS:")
    G = make_biggan(model_name=gan_type)

    state_dict  = G.G.state_dict()
    config = G.G.config

    torch.save(state_dict, "models/"+gan_type+"_pretrained_model.pt")

    with open("models/"+gan_type+"_config.json", "w") as f:
        f.write(config.to_json_string())

