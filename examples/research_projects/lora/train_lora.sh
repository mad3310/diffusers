export MODEL_NAME="/Users/zhoubingzheng/projects/huggingface/model/stable-diffusion-v1-5"
export DATASET_NAME="/Users/zhoubingzheng/projects/huggingface/dataset/fashion-dataset"
export LORA_MODEL_NAME="/Users/zhoubingzheng/projects/huggingface/model/fashion-lora-model"

accelerate launch --mixed_precision="no" train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME --caption_column="text" \
  --resolution=512 --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=100 --checkpointing_steps=5000 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir=$LORA_MODEL_NAME \
  --validation_prompt="fashion chinese man" --report_to="wandb" \
  --use_peft \
  --lora_r=4 --lora_alpha=32 \
  --lora_text_encoder_r=4 --lora_text_encoder_alpha=32