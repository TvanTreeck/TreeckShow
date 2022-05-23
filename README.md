# TreeckShow
 GAN based image synthesis

# How to 

## download TreeckShow repository
- i.e.
  - on https://github.com/TvanTreeck/TreeckShow
  - in upper'ish, right'ish corner 
  - click code -> download ZIP 

## install docker 
https://docs.docker.com/get-docker/

## open terminal

- mac: 
  - cmd + space 
  - type "terminal" 
  - Enter

## go to the project directory
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

- open run.sh with texteditor
  - set 
  ```
    --img_size=84 \
    --img_width=63 \
    --channels=1
  ```
  images will be resampled according to `img_size` and `img_width` 
  `channels=1` for black-white `channels=3` for color
- 
## run
```
bash scripts/run.sh
```