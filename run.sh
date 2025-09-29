MACHINE=h100
MODEL=meta-llama/Llama-3.1-8B-Instruct
MODEL_ID=llama3_8b
TOP_K=128

uv run save_token_dist.py \
    --model_id ${MODEL} \
    --prompt_file prompt.txt \
    --top_k ${TOP_K} \
    --output ${MODEL_ID}_${MACHINE}.json

uv run save_token_dist_dataset.py \
  --model_id ${MODEL} \
  --n_samples 25 \
  --top_k ${TOP_K} \
  --output ${MODEL_ID}_${MACHINE}_mmlupro.jsonl

#uv run compute_kl.py --p_file llama3_8b_h100_mmlupro.jsonl --q_file llama3_8b_4090_mmlupro.jsonl