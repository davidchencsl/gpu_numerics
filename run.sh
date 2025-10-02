# MACHINE=h100
# MODEL=meta-llama/Llama-3.1-8B-Instruct
# MODEL_ID=llama3_8b
# TOP_K=128

# # uv run save_token_dist.py \
# #     --model_id ${MODEL} \
# #     --prompt_file prompt.txt \
# #     --top_k ${TOP_K} \
# #     --output ${MODEL_ID}_${MACHINE}.json

# # uv run save_token_dist_dataset.py \
# #   --model_id ${MODEL} \
# #   --n_samples 25 \
# #   --top_k ${TOP_K} \
# #   --output ${MODEL_ID}_${MACHINE}_mmlupro.json

# uv run compute_kl.py --p_file ${MODEL_ID}_h100_mmlupro.json --q_file ${MODEL_ID}_4090_mmlupro.json --quiet_positions
# uv run compute_kl.py --p_file ${MODEL_ID}_h100.json --q_file ${MODEL_ID}_4090.json --quiet_positions

models=(meta-llama/Llama-3.2-1B-Instruct RedHatAI/Llama-3.2-1B-Instruct-FP8 meta-llama/Llama-3.1-8B-Instruct RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8)

for model in ${models[@]}; do
    uv run benchmark_mmlupro.py --model $model --prompt_json sample_questions.json
done