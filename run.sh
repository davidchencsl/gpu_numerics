# uv run save_token_dist.py \
#     --model_id meta-llama/Llama-3.1-8B \
#     --prompt_file prompt.txt \
#     --top_k 128 \
#     --output llama3_8b_h100.json

uv run save_token_dist_dataset.py \
  --model_id meta-llama/Llama-3.1-8B \
  --n_samples 25 \
  --top_k 128 \
  --output llama3_8b_h100_mmlupro.jsonl

#uv run compute_kl.py --p_file llama3_8b_h100_mmlupro.jsonl --q_file llama3_8b_h100_mmlupro.jsonl