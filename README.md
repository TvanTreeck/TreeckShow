# TreeckShow
- GAN finetuning and image synthesis
- based on https://github.com/lukemelas/pytorch-pretrained-gans/tree/main/pytorch_pretrained_gans/BigGAN

# Goal:
be able to synthesize new images only based on a folder of existing images 

# How to

## download TreeckShow repository
- i.e.
  - on https://github.com/TvanTreeck/TreeckShow
  - in upper'ish, right'ish corner 
  - click code -> download ZIP
  - unzip and place at a location of your choice 

## assemble images
- create a folder called `images` inside the TreeckShow directory 
- place the images inside the folder, which should be bused to train the AI
- NOTE: by default images will be cut out to a quadratic piece and resampled to (3, 128, 128), so we have colored images with bad resolution 


## install docker 
https://docs.docker.com/get-docker/

## open terminal

- mac: 
  - cmd + space 
  - type "terminal" 
  - Enter

## go to the TreeckShow directory
```
cd /path/to/your/TreeckShow
```

## build image 
```
bash scripts/build_image.sh
```

## start container 
```
bash scripts/start_interactive.sh
```

## configure 

- open `scripts/run.s` with any texteditor
  - set 
  ```
    --img_size=128 \
    --img_width=128 \
    --channels=3
  ```
  images will be resampled according to `img_size` and `img_width` 

  `channels=1` for black-white `channels=3` for color

  the current version aims at finetuning an existing model (check: `pytorch-pretrained-gans`), so only this configuration is currently supported

## run
```
bash scripts/run.sh
```