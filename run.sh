python3 prepare_data.py \
  --imgs_path="Images" \
  --img_size=84 \
  --img_width=63 \
  --channels=1   # mit # deactivieren

python gan.py \
  --img_size=84 \
  --img_width=63 \
  --channels=1