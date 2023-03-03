export MODEL_NAME="/Users/zhoubingzheng/projects/huggingface/stable-diffusion-v1-5"
export TRAIN_DIR="/Users/zhoubingzheng/projects/huggingface/fashion-dataset/"
export LOG_DIR="/Users/zhoubingzheng/projects/logs/text-image-train-log"
export PYTORCH_ENABLE_MPS_FALLBACK=1

accelerate launch --mixed_precision="no" train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --image_column="image" \
  --caption_column="text" \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=5 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --logging_dir=$LOG_DIR \
  --report_to="tensorboard" \
  --output_dir="/Users/zhoubingzheng/projects/huggingface/sd-fashion-model"