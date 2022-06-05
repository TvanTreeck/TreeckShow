
echo "DOWNLOAD"
python treeckshow/download_pretrained_model.py 128

echo "PREPARE DATA"
python treeckshow/prepare_data.py \
  --imgs_path="images" \
  --img_size=128 \
  --img_width=128 \
  --channels=3

echo "GENERATE DISCRIMINATOR FEATS"
python treeckshow/generate_discriminator_features.py \
  --model_size=128 \
  --sample_mode=sample

echo "TRAIN DISCRIMINATOR"
python treeckshow/train_discriminator.py \
  --img_size=128 \
  --img_width=128 \
  --channels=3

echo "TRAIN GENERATOR"
python treeckshow/train.py \
  --img_size=128 \
  --img_width=128 \
  --channels=3  \
  --sample_mode=1  # sample für alle classen, classen index (zahl) für eine classe