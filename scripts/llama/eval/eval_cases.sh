#!/bin/bash

#SBATCH --job-name=mgm_eval
#SBATCH --output=mgm_eval.out.%j
#SBATCH --error=mgm_eval.out.%j
#SBATCH --time=2:00:00
#SBATCH --account=scavenger 
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G


CKPT="MGM/MGM-7B"
SPLIT="MMBench_DEV_EN"

CUDA_VISIBLE_DEVICES=0 python -m mgm.eval.model_vqa_cases \
    --model-path ./work_dirs/$CKPT \
    --question-file ./data/MGM-Eval/CVB/$SPLIT.tsv \
    --answers-file ./data/MGM-Eval/CVB/cases/$SPLIT/$CKPT.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1 

# mkdir -p ./data/MGM-Eval/mmbench/answers_upload/$SPLIT

# python scripts/convert_mmbench_for_submission.py \
#     --annotation-file ./data/MGM-Eval/mmbench/$SPLIT.tsv \
#     --result-dir ./data/MGM-Eval/mmbench/answers/$SPLIT \
#     --upload-dir ./data/MGM-Eval/mmbench/answers_upload/$SPLIT \
#     --experiment $CKPT
