#!/bin/bash
CUDA_VISIBLE_DEVICES=2 \
python train.py \
    --model_name_or_path /data/zzc/models/bert-base-uncased \
    --train_file data/wiki1m_for_simcse.txt \
    --output_dir runs/scratch-listmle-bert-base-uncased \
    --num_train_epochs 1 \
    --per_device_train_batch_size 128 \
    --learning_rate 3e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    --first_teacher_name_or_path /data/zzc/models/diffcse-bert-base-uncased-sts \
    --second_teacher_name_or_path /data/zzc/models/unsup-simcse-bert-large-uncased \
    --distillation_loss listmle \
    --alpha_ 0.33 \
    --beta_ 1.0 \
    --gamma_ 1.0 \
    --tau2 0.05


CUDA_VISIBLE_DEVICES=2 \
python evaluation.py \
    --model_name_or_path runs/scratch-listmle-bert-base-uncased \
    --pooler cls_before_pooler \
    --task_set sts \
    --mode test
