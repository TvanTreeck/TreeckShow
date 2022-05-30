python treeckshow/download_pretrained_model.py

python treeckshow/prepare_data.py \
  --imgs_path="images" \
  --img_size=128 \
  --img_width=128 \
  --channels=3

python treeckshow/generate_discriminator_features.py \
  --sample_mode="1" # sample für alle classen, classen index (zahl) für eine classe

python treeckshow/train_discriminator.py

python treeckshow/train.py \
  --sample_mode="1" \
  --img_size=128 \
  --img_width=128 \
  --channels=3 
