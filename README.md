# MMLU-Pro Benchmark Scripts

Two scripts for evaluating language models on MMLU-Pro dataset:
1. `generate_question_sample.py` - Randomly sample questions and generate prompts
2. `benchmark_mmlupro.py` - Evaluate models and compute accuracy

## Workflow

### Step 1: Generate Question Sample with Prompts

First, create a sample of MMLU-Pro questions (e.g., 10% random sample):

```bash
python generate_question_sample.py \
    --percentage 10 \
    --output sample_10pct.json \
    --k_shot 5 \
    --seed 42
```

This creates a JSON file with:
- `question_id`: Unique question identifier
- `category`: Subject category (e.g., "biology", "economics")
- `prompt`: Full chain-of-thought prompt with few-shot examples
- `answer`: Ground truth answer letter
- `question`, `options`, `answer_index`: Question details

**Output example:**
```json
[
  {
    "question_id": "...",
    "category": "biology",
    "prompt": "The following are multiple choice questions...",
    "prompt_ids": [123, 456, 789, ...],  // Optional: only if --model is specified
    "question": "What is photosynthesis?",
    "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
    "answer": "B",
    "answer_index": 1
  },
  ...
]
```

**Note:** The `prompt_ids` field (list of token integers) is only included when you specify the `--model` argument. This is useful for pre-tokenizing prompts for faster inference.

### Step 2: Evaluate Model on Sampled Questions

Run the benchmark using the generated prompts with vLLM:

```bash
python benchmark_mmlupro.py \
    --model meta-llama/Llama-2-7b-hf \
    --prompt_json sample_10pct.json
```

The script will:
1. Load the model with vLLM (fast inference engine)
2. Batch process all questions in each category
3. Extract answers from model outputs
4. Compute accuracy (overall and per-category)
5. Save detailed results and summary

## `generate_question_sample.py` Usage

### Arguments

- `--percentage` (required): Percentage of questions to sample (0-100)
- `--output`: Output JSON file (default: `sample_questions.json`)
- `--k_shot`: Number of few-shot examples in prompts (default: 5)
- `--seed`: Random seed for reproducibility (default: 42)
- `--model`: Model ID for tokenization (optional). If provided, generates `prompt_ids` as list of token integers

### Examples

```bash
# Sample 5% of questions (text prompts only)
python generate_question_sample.py --percentage 5 --output sample_5pct.json

# Sample 20% with 3-shot prompts
python generate_question_sample.py --percentage 20 --k_shot 3 --output sample_20pct_3shot.json

# Sample with tokenization (includes prompt_ids as list of integers)
python generate_question_sample.py \
    --percentage 10 \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --output sample_10pct_with_ids.json

# Use different random seed
python generate_question_sample.py --percentage 10 --seed 123 --output sample_seed123.json
```

## `benchmark_mmlupro.py` Usage

**Note: This script uses vLLM for fast batched inference. Make sure you have vLLM installed:**
```bash
pip install vllm
```

### Mode 1: Use Pre-generated Prompts (Recommended)

```bash
python benchmark_mmlupro.py \
    --model MODEL_NAME \
    --prompt_json sample_questions.json
```

### Mode 2: Generate Prompts On-the-Fly

```bash
# Evaluate all categories
python benchmark_mmlupro.py \
    --model MODEL_NAME \
    --selected_subjects all \
    --ntrain 5

# Evaluate specific categories
python benchmark_mmlupro.py \
    --model MODEL_NAME \
    --selected_subjects "biology,economics" \
    --ntrain 5
```

### Arguments

- `--model` / `-m` (required): HuggingFace model ID or local path
- `--prompt_json`: JSON file with pre-generated prompts (recommended for faster setup)
- `--ntrain` / `-k`: Number of few-shot examples (default: 5)
- `--selected_subjects` / `-sub`: Categories to evaluate (default: "all"). Ignored if `--prompt_json` is used
- `--save_dir` / `-s`: Directory to save results (default: "results")
- `--global_record_file` / `-grf`: CSV file to record results (default: "eval_record_collection.csv")

### Evaluation Parameters

- **vLLM batched inference**: Automatically batches all questions for maximum throughput
- **Greedy decoding**: Uses `temperature=0` for deterministic outputs
- **Max new tokens**: 128 (optimized for efficiency)
- **Max model length**: 4096
- **Seed**: 42 (for reproducibility)

## Output Format

The benchmark script outputs detailed results with token usage information:

### Console Output

During evaluation, the script logs:
- Progress for each category
- Token usage per category (prompt tokens, output tokens, total tokens)
- Accuracy statistics per category
- Overall summary

Summary file saved in `results/summary/MODEL_NAME-CoT-SUBJECTS_TIMESTAMP_summary.txt` contains:

```
------category level sta------
Average accuracy 0.6667 - biology
Average accuracy 0.5000 - computer science
Average accuracy 0.7000 - economics
...

------average acc sta------
Average accuracy: 0.6250
```

**Log output includes:**
- Batch inference timing
- Token usage per category: `Token usage - Prompt: 12340, Output: 1560, Total: 13900`
- Per-category accuracy: `Accuracy: 0.6667, Correct: 10, Wrong: 5`

### JSON Output File

Each category's results are saved separately in `results/MODEL_NAME/CoT/SUBJECTS/CATEGORY.json`:

```json
[
  {
    "question_id": "...",
    "category": "biology",
    "question": "What is photosynthesis?",
    "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
    "answer": "B",
    "answer_index": 1,
    "pred": "B",
    "model_outputs": "Let's think step by step. Photosynthesis is the process... Therefore, the answer is (B).",
    "prompt_tokens": 1234,
    "output_tokens": 156,
    "total_tokens": 1390
  },
  {
    "question_id": "...",
    "category": "biology",
    "question": "...",
    "options": [...],
    "answer": "C",
    "answer_index": 2,
    "pred": "C",
    "model_outputs": "Let's think step by step...",
    "prompt_tokens": 1189,
    "output_tokens": 203,
    "total_tokens": 1392
  },
  ...
]
```

**New fields in output:**
- `model_outputs`: Full text response from the model
- `prompt_tokens`: Number of tokens in the input prompt
- `output_tokens`: Number of tokens generated by the model
- `total_tokens`: Sum of prompt and output tokens

## Complete Example Workflow

```bash
# Step 1: Generate 10% sample
python generate_question_sample.py \
    --percentage 10 \
    --output mmlu_pro_10pct.json \
    --k_shot 5

# Step 2: Evaluate Model A with vLLM (fast!)
python benchmark_mmlupro.py \
    --model meta-llama/Llama-2-7b-hf \
    --prompt_json mmlu_pro_10pct.json

# Step 3: Evaluate Model B (using same prompts for fair comparison)
python benchmark_mmlupro.py \
    --model mistralai/Mistral-7B-v0.1 \
    --prompt_json mmlu_pro_10pct.json
```

Results will be saved in:
- `results/MODEL_NAME/CoT/SUBJECTS/CATEGORY.json` - Per-category results
- `results/summary/MODEL_NAME-CoT-SUBJECTS_TIMESTAMP_summary.txt` - Summary with accuracy stats
- `results/log/` - Detailed logs

## Benefits of Two-Step Approach

1. **Speed**: vLLM's batched inference is significantly faster than transformers
2. **Consistency**: Same prompts used for all models (fair comparison)
3. **Reproducibility**: Fixed random seed ensures same questions
4. **Flexibility**: Can evaluate subset without full dataset load each time
5. **Caching**: Pre-generated prompts can be reused across model evaluations
6. **Efficiency**: vLLM handles optimal batching automatically
7. **Simplicity**: No need to manage workers or batch sizes manually

## Notes

- **vLLM required**: Fast inference engine optimized for LLMs (`pip install vllm`)
- Models are loaded with vLLM which automatically optimizes for your GPU
- Random seed (42) ensures reproducible sampling and generation
- For questions where answer cannot be extracted, a random guess is made
- Accuracy is computed both per-category and overall
- vLLM automatically batches requests for maximum throughput
- All inference uses greedy decoding (temperature=0) for deterministic results

## Requirements

```bash
# For benchmark script (vLLM-based)
pip install vllm torch transformers datasets tqdm

# For question sampling script
pip install datasets tqdm
```
