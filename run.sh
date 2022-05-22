python download_pretrained_model.py

python prepare_data.py \
  --imgs_path="images" \
  --img_size=128 \
  --img_width=128 \
  --channels=3

python train.py \
  --img_size=128 \
  --img_width=128 \
  --channels=3