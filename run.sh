python3 prepare_data.py \
  --imgs_path="images" \
  --img_size=84 \
  --img_width=63 \
  --channels=1   

python gan.py \
  --img_size=84 \
  --img_width=63 \
  --channels=1