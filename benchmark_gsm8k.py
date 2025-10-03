#!/usr/bin/env python3
"""
GSM8K Evaluation Script using vLLM for fast inference.

Usage:
  uv run benchmark_gsm8k.py --model MODEL_NAME
  
Features:
  - Fast batched inference with vLLM
  - Accuracy reporting
  - Greedy decoding with temperature=0
  - Chain-of-thought prompting (8-shot)
"""

import json
import argparse
import os
import torch
import numpy as np
import random
import transformers
import time
import re
from vllm import LLM, SamplingParams
from tqdm import tqdm
import logging
import sys
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

max_model_length = 4096
max_new_tokens = 512  # GSM8K needs more tokens for reasoning

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"   # makes GEMMs repeatable

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"   # turn off multi-process engine


def load_gsm8k():
    """Load GSM8K dataset from HuggingFace."""
    logging.info("Loading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main")
    train_data = dataset["train"]
    test_data = dataset["test"]
    
    # Convert to list of dicts for easier handling
    train_df = [{"question": item["question"], "answer": item["answer"]} for item in train_data]
    test_df = [{"question": item["question"], "answer": item["answer"]} for item in test_data]
    
    logging.info(f"Loaded {len(train_df)} training examples and {len(test_df)} test examples")
    return train_df, test_df


def load_model():
    """Initialize vLLM model and sampling parameters."""
    llm = LLM(model=args.model,
                tensor_parallel_size=1,
                max_model_len=max_model_length,
                trust_remote_code=True,
                )
    sampling_params = SamplingParams(temperature=0, 
                                     max_tokens=max_new_tokens,
                                     seed=SEED)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, token=os.getenv("HF_TOKEN"))
    return (llm, sampling_params), tokenizer


def extract_answer_number(text):
    """
    Extract the numerical answer from the model's output.
    Looks for patterns like:
    - "#### 123"
    - "The answer is 123"
    - Last number in the text
    """
    # GSM8K format: look for #### followed by the answer
    pattern = r"####\s*([-+]?[\d,]*\.?\d+)"
    match = re.search(pattern, text)
    if match:
        answer_str = match.group(1).replace(",", "")
        try:
            return float(answer_str)
        except ValueError:
            pass
    
    # Look for "answer is X" pattern
    pattern = r"[Tt]he answer is:?\s*([-+]?[\d,]*\.?\d+)"
    match = re.search(pattern, text)
    if match:
        answer_str = match.group(1).replace(",", "")
        try:
            return float(answer_str)
        except ValueError:
            pass
    
    # Look for boxed answer
    pattern = r"\\boxed\{([-+]?[\d,]*\.?\d+)\}"
    match = re.search(pattern, text)
    if match:
        answer_str = match.group(1).replace(",", "")
        try:
            return float(answer_str)
        except ValueError:
            pass
    
    # Fallback: extract last number in the text
    numbers = re.findall(r"[-+]?[\d,]*\.?\d+", text)
    if numbers:
        # Take the last number, removing commas
        answer_str = numbers[-1].replace(",", "")
        try:
            return float(answer_str)
        except ValueError:
            pass
    
    logging.warning(f"Could not extract answer from: {text[:100]}...")
    return None


def extract_ground_truth(answer_str):
    """Extract the numerical answer from GSM8K's answer format."""
    # GSM8K answers are in format: "explanation #### 123"
    pattern = r"####\s*([-+]?[\d,]*\.?\d+)"
    match = re.search(pattern, answer_str)
    if match:
        answer = match.group(1).replace(",", "")
        return float(answer)
    
    # Fallback: try to extract any number
    numbers = re.findall(r"[-+]?[\d,]*\.?\d+", answer_str)
    if numbers:
        return float(numbers[-1].replace(",", ""))
    
    logging.error(f"Could not extract ground truth from: {answer_str}")
    return None


def format_few_shot_example(question, answer):
    """Format a few-shot example."""
    # Extract the numerical answer for clean display
    ground_truth = extract_ground_truth(answer)
    
    prompt = f"Question: {question}\n"
    prompt += f"Answer: Let's think step by step.\n{answer}\n\n"
    return prompt


def generate_cot_prompt(train_df, question, k=8):
    """
    Generate chain-of-thought prompt with k few-shot examples.
    
    Args:
        train_df: Training examples for few-shot learning
        question: The test question to answer
        k: Number of few-shot examples
    
    Returns:
        Formatted prompt string
    """
    # System prompt
    prompt = (
        "Solve the following grade school math problems. "
        "Think step by step and show your work. "
        "End your answer with '#### ' followed by the final numerical answer.\n\n"
    )
    
    # Add k few-shot examples
    few_shot_examples = train_df[:k]
    for example in few_shot_examples:
        prompt += format_few_shot_example(example["question"], example["answer"])
    
    # Add the current question
    prompt += f"Question: {question}\n"
    prompt += "Answer: Let's think step by step.\n"
    
    return prompt


def batch_inference(llm, sampling_params, inference_batch):
    """Run batched inference on a list of prompts."""
    start = time.time()
    outputs = llm.generate(inference_batch, sampling_params)
    logging.info(f"{len(inference_batch)} batch size costing time: {time.time() - start:.2f}s")
    
    response_batch = []
    pred_batch = []
    token_info_batch = []
    
    for output in outputs:
        generated_text = output.outputs[0].text
        response_batch.append(generated_text)
        
        # Extract predicted answer
        pred = extract_answer_number(generated_text)
        pred_batch.append(pred)
        
        # Extract token counts
        prompt_tokens = len(output.prompt_token_ids)
        output_tokens = len(output.outputs[0].token_ids)
        token_info = {
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": prompt_tokens + output_tokens
        }
        token_info_batch.append(token_info)
    
    return pred_batch, response_batch, token_info_batch


def save_results(res, output_path):
    """Save results to JSON and calculate accuracy."""
    # Save results to JSON
    with open(output_path, "w") as fo:
        json.dump(res, fo, indent=2)
    
    # Calculate accuracy and token statistics
    correct = 0
    wrong = 0
    no_answer = 0
    total_prompt_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    
    for item in res:
        # Token counts
        if "prompt_tokens" in item:
            total_prompt_tokens += item["prompt_tokens"]
            total_output_tokens += item["output_tokens"]
            total_tokens += item["total_tokens"]
        
        # Accuracy calculation
        if item["pred"] is None:
            no_answer += 1
            wrong += 1
        elif abs(item["pred"] - item["ground_truth"]) < 1e-4:  # Account for floating point errors
            correct += 1
        else:
            wrong += 1
    
    total = correct + wrong
    accuracy = correct / total if total > 0 else 0.0
    
    # Log statistics
    logging.info(f"Correct: {correct}, Wrong: {wrong}, No Answer: {no_answer}, Total: {total}")
    logging.info(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    logging.info(f"Token usage - Prompt: {total_prompt_tokens}, Output: {total_output_tokens}, Total: {total_tokens}")
    
    return accuracy, correct, wrong


@torch.no_grad()
def eval_gsm8k(model, tokenizer, train_df, test_df, output_path):
    """Evaluate on GSM8K test set."""
    llm, sampling_params = model
    
    logging.info(f"Evaluating on {len(test_df)} test questions...")
    
    # Generate prompts for all test questions
    inference_batches = []
    for i in tqdm(range(len(test_df)), desc="Generating prompts"):
        question = test_df[i]["question"]
        k = args.ntrain
        
        # Check prompt length and adjust k if needed
        prompt_length_ok = False
        while not prompt_length_ok and k >= 0:
            prompt = generate_cot_prompt(train_df, question, k=k)
            inputs = tokenizer(prompt, return_tensors="pt")
            length = len(inputs["input_ids"][0])
            
            if length < max_model_length - max_new_tokens:
                prompt_length_ok = True
            else:
                k -= 1
                logging.warning(f"Prompt too long, reducing k to {k}")
        
        inference_batches.append(prompt)
    
    # Run batched inference
    pred_batch, response_batch, token_info_batch = batch_inference(llm, sampling_params, inference_batches)
    
    # Compile results
    res = []
    for j, item in enumerate(test_df):
        ground_truth = extract_ground_truth(item["answer"])
        
        result = {
            "question_id": j,
            "question": item["question"],
            "ground_truth": ground_truth,
            "ground_truth_raw": item["answer"],
            "pred": pred_batch[j],
            "model_outputs": response_batch[j],
            "prompt_tokens": token_info_batch[j]["prompt_tokens"],
            "output_tokens": token_info_batch[j]["output_tokens"],
            "total_tokens": token_info_batch[j]["total_tokens"]
        }
        res.append(result)
    
    # Save and evaluate
    accuracy, correct, wrong = save_results(res, output_path)
    
    return accuracy, correct, wrong


def main():
    """Main evaluation function."""
    # Load model
    model, tokenizer = load_model()
    
    # Load dataset
    train_df, test_df = load_gsm8k()
    
    # Create output directory
    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir)
    
    # Run evaluation
    output_path = os.path.join(save_result_dir, "gsm8k_results.json")
    accuracy, correct, wrong = eval_gsm8k(model, tokenizer, train_df, test_df, output_path)
    
    # Save summary
    with open(summary_path, 'a') as f:
        f.write("\n" + "="*80 + "\n")
        f.write("GSM8K EVALUATION RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Few-shot examples (k): {args.ntrain}\n")
        f.write(f"Correct: {correct}\n")
        f.write(f"Wrong: {wrong}\n")
        f.write(f"Total: {correct + wrong}\n")
        f.write(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write("="*80 + "\n")
    
    logging.info(f"\nResults saved to: {output_path}")
    logging.info(f"Summary saved to: {summary_path}")
    logging.info(f"\nFinal Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on GSM8K dataset")
    parser.add_argument("--ntrain", "-k", type=int, default=8,
                        help="Number of few-shot examples (default: 8)")
    parser.add_argument("--model", "-m", type=str, required=True,
                        help="Model name or path")
    parser.add_argument("--save_dir", "-s", type=str, default="results",
                        help="Directory to save results")
    
    args = parser.parse_args()
    
    # Setup directories and paths
    os.makedirs(args.save_dir, exist_ok=True)
    model_name = args.model.split("/")[-1]
    save_result_dir = os.path.join(args.save_dir, f"gsm8k_{model_name}")
    os.makedirs(save_result_dir, exist_ok=True)
    
    # Setup logging
    timestamp = time.time()
    time_str = time.strftime('%m-%d_%H-%M', time.localtime(timestamp))
    log_dir = os.path.join(args.save_dir, "log")
    os.makedirs(log_dir, exist_ok=True)
    summary_dir = os.path.join(args.save_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    summary_filename = f"gsm8k_{model_name}_{time_str}_summary.txt"
    summary_path = os.path.join(summary_dir, summary_filename)
    log_filename = f"gsm8k_{model_name}_{time_str}_logfile.log"
    log_path = os.path.join(log_dir, log_filename)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info(f"Starting GSM8K evaluation with model: {args.model}")
    logging.info(f"Using {args.ntrain}-shot prompting")
    
    main()

