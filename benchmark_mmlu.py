#!/usr/bin/env python3
"""
MMLU (Massive Multitask Language Understanding) Evaluation Script using vLLM for fast inference.

Usage:
  # Evaluate on all subjects
  uv run benchmark_mmlu.py --model MODEL_NAME --selected_subjects all
  
  # Evaluate on specific subjects
  uv run benchmark_mmlu.py --model MODEL_NAME --selected_subjects "high_school_physics,college_chemistry"
  
Features:
  - Fast batched inference with vLLM
  - Support for 57 subjects across STEM, humanities, social sciences, etc.
  - Accuracy reporting by subject and overall
  - Greedy decoding with temperature=0
  - 5-shot prompting by default
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
import csv
from collections import defaultdict

load_dotenv()

choices = ["A", "B", "C", "D"]
max_model_length = 4096
max_new_tokens = 128

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

# MMLU subject categories
SUBJECT_CATEGORIES = {
    "STEM": [
        "abstract_algebra", "anatomy", "astronomy", "college_biology", 
        "college_chemistry", "college_computer_science", "college_mathematics",
        "college_physics", "computer_security", "conceptual_physics",
        "electrical_engineering", "elementary_mathematics", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science", "high_school_mathematics",
        "high_school_physics", "high_school_statistics", "machine_learning"
    ],
    "Humanities": [
        "formal_logic", "high_school_european_history", "high_school_us_history",
        "high_school_world_history", "international_law", "jurisprudence",
        "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy",
        "prehistory", "professional_law", "world_religions"
    ],
    "Social Sciences": [
        "econometrics", "high_school_geography", "high_school_government_and_politics",
        "high_school_macroeconomics", "high_school_microeconomics",
        "high_school_psychology", "human_sexuality", "professional_psychology",
        "public_relations", "security_studies", "sociology", "us_foreign_policy"
    ],
    "Other": [
        "business_ethics", "clinical_knowledge", "college_medicine",
        "global_facts", "human_aging", "management", "marketing",
        "medical_genetics", "miscellaneous", "nutrition", "professional_accounting",
        "professional_medicine", "virology"
    ]
}


def get_all_subjects():
    """Get list of all MMLU subjects."""
    all_subjects = []
    for category_subjects in SUBJECT_CATEGORIES.values():
        all_subjects.extend(category_subjects)
    return sorted(all_subjects)


def load_mmlu(subject):
    """Load MMLU dataset for a specific subject."""
    try:
        dataset = load_dataset("cais/mmlu", subject)
        dev_df = dataset["dev"]  # Few-shot examples
        test_df = dataset["test"]  # Test set
        val_df = dataset["validation"] if "validation" in dataset else dev_df  # Validation set
        
        # Convert to list of dicts
        dev_data = [dict(item) for item in dev_df]
        test_data = [dict(item) for item in test_df]
        val_data = [dict(item) for item in val_df]
        
        return dev_data, test_data, val_data
    except Exception as e:
        logging.error(f"Error loading subject {subject}: {e}")
        return [], [], []


def load_model():
    """Initialize vLLM model and sampling parameters."""
    llm = LLM(model=args.model,
                tensor_parallel_size=1,
                max_model_len=max_model_length,
                trust_remote_code=True,
                )
    sampling_params = SamplingParams(temperature=0, 
                                     max_tokens=max_new_tokens,
                                     seed=SEED,
                                     stop=["Question:", "\n\n"])
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, token=os.getenv("HF_TOKEN"))
    return (llm, sampling_params), tokenizer


def format_subject(subject):
    """Format subject name for display."""
    return subject.replace("_", " ").title()


def format_example(question, choices_list, answer_idx=None, include_answer=True):
    """
    Format a single MMLU example.
    
    Args:
        question: Question text
        choices_list: List of 4 choice strings
        answer_idx: Index of correct answer (0-3)
        include_answer: Whether to include the answer
    
    Returns:
        Formatted string
    """
    prompt = f"Question: {question}\n"
    for i, choice in enumerate(choices_list):
        prompt += f"{choices[i]}. {choice}\n"
    
    if include_answer and answer_idx is not None:
        prompt += f"Answer: {choices[answer_idx]}\n\n"
    else:
        prompt += "Answer:"
    
    return prompt


def generate_few_shot_prompt(dev_data, question, choices_list, k=5):
    """
    Generate few-shot prompt with k examples from dev set.
    
    Args:
        dev_data: Development/few-shot examples
        question: Test question
        choices_list: List of choices for test question
        k: Number of few-shot examples
    
    Returns:
        Formatted prompt string
    """
    # Start with instruction
    prompt = "The following are multiple choice questions (with answers).\n\n"
    
    # Add k few-shot examples from dev set
    num_examples = min(k, len(dev_data))
    for i in range(num_examples):
        example = dev_data[i]
        prompt += format_example(
            example["question"],
            example["choices"],
            example["answer"],
            include_answer=True
        )
    
    # Add the test question
    prompt += format_example(question, choices_list, include_answer=False)
    
    return prompt


def extract_answer(text):
    """
    Extract the answer choice (A, B, C, or D) from model output.
    
    Args:
        text: Model's generated text
    
    Returns:
        Predicted choice letter or None if extraction fails
    """
    # Look for single letter answer at the start
    match = re.match(r'^\s*([A-D])', text)
    if match:
        return match.group(1)
    
    # Look for "Answer: X" or "answer is X" patterns
    pattern = r'[Aa]nswer:?\s*([A-D])'
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    
    # Look for "The answer is (X)" or "answer is X"
    pattern = r'[Aa]nswer\s+is\s+\(?([A-D])\)?'
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    
    # Look for standalone letter choice
    pattern = r'\b([A-D])\b'
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    
    logging.warning(f"Could not extract answer from: {text[:100]}...")
    return None


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
        pred = extract_answer(generated_text)
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
            wrong += 1
        elif item["pred"] == item["answer"]:
            correct += 1
        else:
            wrong += 1
    
    total = correct + wrong
    accuracy = correct / total if total > 0 else 0.0
    
    # Log statistics
    if total_tokens > 0:
        logging.info(f"Token usage - Prompt: {total_prompt_tokens}, Output: {total_output_tokens}, Total: {total_tokens}")
    
    return accuracy, correct, wrong


@torch.no_grad()
def eval_subject(subject, model, tokenizer, dev_data, test_data, output_path):
    """Evaluate on a single MMLU subject."""
    llm, sampling_params = model
    
    logging.info(f"Evaluating subject: {subject}")
    
    if len(test_data) == 0:
        logging.warning(f"No test data for subject: {subject}")
        return 0.0, 0, 0
    
    # Generate prompts for all test questions
    inference_batches = []
    for i in tqdm(range(len(test_data)), desc=f"Generating prompts for {subject}"):
        item = test_data[i]
        k = args.ntrain
        
        # Check prompt length and adjust k if needed
        prompt_length_ok = False
        prompt = None
        while not prompt_length_ok and k >= 0:
            prompt = generate_few_shot_prompt(dev_data, item["question"], item["choices"], k=k)
            inputs = tokenizer(prompt, return_tensors="pt")
            length = len(inputs["input_ids"][0])
            
            if length < max_model_length - max_new_tokens:
                prompt_length_ok = True
            else:
                k -= 1
                if k >= 0:
                    logging.warning(f"Prompt too long, reducing k to {k}")
        
        inference_batches.append(prompt)
    
    # Run batched inference
    pred_batch, response_batch, token_info_batch = batch_inference(llm, sampling_params, inference_batches)
    
    # Compile results
    res = []
    for j, item in enumerate(test_data):
        result = {
            "question_id": j,
            "subject": subject,
            "question": item["question"],
            "choices": item["choices"],
            "answer": choices[item["answer"]],
            "answer_index": item["answer"],
            "pred": pred_batch[j],
            "model_outputs": response_batch[j],
            "prompt_tokens": token_info_batch[j]["prompt_tokens"],
            "output_tokens": token_info_batch[j]["output_tokens"],
            "total_tokens": token_info_batch[j]["total_tokens"]
        }
        res.append(result)
    
    # Save and evaluate
    accuracy, correct, wrong = save_results(res, output_path)
    logging.info(f"Subject {subject} - Accuracy: {accuracy:.4f}, Correct: {correct}, Wrong: {wrong}")
    
    return accuracy, correct, wrong


def args_generate_path(input_args):
    """Generate path components from arguments."""
    model_name = input_args.model.split("/")[-1]
    subjects = "all" if input_args.selected_subjects == "all" else input_args.selected_subjects.replace(",", "-").replace(" ", "_")
    return [model_name, subjects]


def main():
    """Main evaluation function."""
    # Load model
    model, tokenizer = load_model()
    
    # Create output directory
    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir)
    
    # Determine which subjects to evaluate
    all_subjects = get_all_subjects()
    
    if args.selected_subjects == "all":
        selected_subjects = all_subjects
    else:
        # Parse comma-separated list of subjects
        requested = [s.strip() for s in args.selected_subjects.split(",")]
        selected_subjects = []
        for subject in all_subjects:
            for req in requested:
                # Allow partial matching
                if req.replace(" ", "_").lower() in subject.replace(" ", "_").lower():
                    if subject not in selected_subjects:
                        selected_subjects.append(subject)
        
        if not selected_subjects:
            logging.error(f"No matching subjects found for: {args.selected_subjects}")
            logging.info(f"Available subjects: {', '.join(all_subjects[:10])}... (and {len(all_subjects)-10} more)")
            return
    
    selected_subjects = sorted(selected_subjects)
    logging.info(f"Selected {len(selected_subjects)} subjects:\n" + "\n".join(selected_subjects))
    print(f"\nEvaluating {len(selected_subjects)} subjects...")
    
    # Write header to summary file
    with open(summary_path, 'a') as f:
        f.write("\n" + "="*80 + "\n")
        f.write("MMLU EVALUATION RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Few-shot examples (k): {args.ntrain}\n")
        f.write(f"Subjects: {len(selected_subjects)}\n")
        f.write("="*80 + "\n")
        f.write("\n------Subject Level Results------\n")
    
    # Evaluate each subject
    sta_dict = {}
    category_stats = defaultdict(lambda: {"correct": 0, "wrong": 0})
    
    for subject in selected_subjects:
        # Load data for this subject
        dev_data, test_data, val_data = load_mmlu(subject)
        
        if len(test_data) == 0:
            logging.warning(f"Skipping {subject} - no test data")
            continue
        
        # Run evaluation
        output_path = os.path.join(save_result_dir, f"{subject}.json")
        accuracy, correct, wrong = eval_subject(subject, model, tokenizer, dev_data, test_data, output_path)
        
        # Store results
        sta_dict[subject] = {
            "correct": correct,
            "wrong": wrong,
            "accuracy": accuracy
        }
        
        # Update category stats
        for category, subjects_in_category in SUBJECT_CATEGORIES.items():
            if subject in subjects_in_category:
                category_stats[category]["correct"] += correct
                category_stats[category]["wrong"] += wrong
                break
        
        # Write to summary
        with open(summary_path, 'a') as f:
            f.write(f"{subject}: {accuracy:.4f} ({correct}/{correct+wrong})\n")
    
    # Calculate overall statistics
    total_correct = sum(v["correct"] for v in sta_dict.values())
    total_wrong = sum(v["wrong"] for v in sta_dict.values())
    total_accuracy = total_correct / (total_correct + total_wrong) if (total_correct + total_wrong) > 0 else 0.0
    
    # Calculate average accuracy (macro-average across subjects)
    avg_accuracy = np.mean([v["accuracy"] for v in sta_dict.values()]) if sta_dict else 0.0
    
    # Write category and overall statistics
    with open(summary_path, 'a') as f:
        f.write("\n------Category Level Results------\n")
        for category in sorted(category_stats.keys()):
            stats = category_stats[category]
            total = stats["correct"] + stats["wrong"]
            if total > 0:
                cat_accuracy = stats["correct"] / total
                f.write(f"{category}: {cat_accuracy:.4f} ({stats['correct']}/{total})\n")
        
        f.write("\n------Overall Results------\n")
        f.write(f"Total Correct: {total_correct}\n")
        f.write(f"Total Wrong: {total_wrong}\n")
        f.write(f"Total Questions: {total_correct + total_wrong}\n")
        f.write(f"Overall Accuracy (Micro): {total_accuracy:.4f} ({total_accuracy*100:.2f}%)\n")
        f.write(f"Average Accuracy (Macro): {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)\n")
        f.write("="*80 + "\n")
    
    # Write to global record file
    with open(global_record_file, 'a', newline='') as file:
        writer = csv.writer(file)
        record = args_generate_path(args) + [time_str, total_accuracy, avg_accuracy]
        writer.writerow(record)
    
    logging.info(f"\n{'='*80}")
    logging.info(f"EVALUATION COMPLETE")
    logging.info(f"{'='*80}")
    logging.info(f"Overall Accuracy (Micro): {total_accuracy:.4f} ({total_accuracy*100:.2f}%)")
    logging.info(f"Average Accuracy (Macro): {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
    logging.info(f"Results saved to: {save_result_dir}")
    logging.info(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on MMLU dataset")
    parser.add_argument("--ntrain", "-k", type=int, default=5,
                        help="Number of few-shot examples (default: 5)")
    parser.add_argument("--selected_subjects", "-sub", type=str, default="all",
                        help='Subjects to evaluate: "all" or comma-separated list')
    parser.add_argument("--model", "-m", type=str, required=True,
                        help="Model name or path")
    parser.add_argument("--save_dir", "-s", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--global_record_file", "-grf", type=str,
                        default="eval_record_collection.csv",
                        help="CSV file to append results to")
    
    args = parser.parse_args()
    
    # Setup directories and paths
    os.makedirs(args.save_dir, exist_ok=True)
    global_record_file = args.global_record_file
    save_result_dir = os.path.join(args.save_dir, "/".join(args_generate_path(args)))
    os.makedirs(save_result_dir, exist_ok=True)
    
    # Setup logging
    timestamp = time.time()
    time_str = time.strftime('%m-%d_%H-%M', time.localtime(timestamp))
    file_prefix = "-".join(args_generate_path(args))
    log_dir = os.path.join(args.save_dir, "log")
    os.makedirs(log_dir, exist_ok=True)
    summary_dir = os.path.join(args.save_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    summary_filename = f"{file_prefix}_{time_str}_summary.txt"
    summary_path = os.path.join(summary_dir, summary_filename)
    log_filename = f"{file_prefix}_{time_str}_logfile.log"
    log_path = os.path.join(log_dir, log_filename)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info(f"Starting MMLU evaluation with model: {args.model}")
    logging.info(f"Using {args.ntrain}-shot prompting")
    
    main()

