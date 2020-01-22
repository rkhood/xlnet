# Fine-tuning XLNet

# directories
TASK_NAME="eg_sts-b"
DATA_DIR="path_to_glue_data"
OUTPUT_DIR="path_to_output"
MODEL_DIR="path_to_tuned_model"
PRETRAIN_DIR="path_to_pretrained_model"


# fine-tune xlnet
sh -c "CUDA_VISIBLE_DEVICES=0 python run_classifier.py \
    --do_train=True \
    --do_eval=False \
    --task_name=$TASK_NAME \
    --data_dir=$DATA_DIR \
    --output_dir=$OUTPUT_DIR \
    --model_dir=$MODEL_DIR \
    --uncased=False \
    --spiece_model_file=$PRETRAIN_DIR/xlnet_cased_L-24_H-1024_A-16/spiece.model \
    --model_config_path=$PRETRAIN_DIR/xlnet_cased_L-24_H-1024_A-16/xlnet_config.json \
    --init_checkpoint=$PRETRAIN_DIR/xlnet_cased_L-24_H-1024_A-16/xlnet_model.ckpt \
    --max_seq_length=128 \
    --train_batch_size=8 \
    --num_hosts=1 \
    --num_core_per_host=1 \
    --learning_rate=5e-5 \
    --train_steps=4800 \
    --warmup_steps=120 \
    --save_steps=600 \
    --is_regression=True"


# evaluate fine-tuned model
sh -c "CUDA_VISIBLE_DEVICES=0 python run_classifier.py \
    --do_train=False \
    --do_eval=True \
    --task_name=$TASK_NAME \
    --data_dir=$DATA_DIR \
    --output_dir=$OUTPUT_DIR \
    --model_dir=$MODEL_DIR \
    --uncased=False \
    --spiece_model_file=$PRETRAIN_DIR/xlnet_cased_L-24_H-1024_A-16/spiece.model \
    --model_config_path=$PRETRAIN_DIR/xlnet_cased_L-24_H-1024_A-16/xlnet_config.json \
    --max_seq_length=128 \
    --eval_batch_size=8 \
    --num_hosts=1 \
    --num_core_per_host=1 \
    --eval_all_ckpt=True \
    --is_regression=True"
