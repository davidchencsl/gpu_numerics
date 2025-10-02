#!/usr/bin/env python3
"""
Script to randomly sample MMLU-Pro questions and generate prompts.
Outputs to JSON with question_id, category, prompt, and optionally prompt_ids for each question.

Usage: 
  # Without tokenization (text prompts only)
  uv run generate_question_sample.py --percentage 100 --output sample_questions.json
  
  # With tokenization (includes prompt_ids as list of integers)
  uv run generate_question_sample.py --percentage 100 --output sample_questions.json --model meta-llama/Llama-3.2-1B-Instruct
"""

import json
import argparse
import os
import random
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import transformers
from dotenv import load_dotenv

load_dotenv()

# Answer choices
CHOICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]

# Set seeds for reproducibility
SEED = 42


def set_seed(seed):
    """Set random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_mmlu_pro():
    """Load MMLU-Pro dataset from HuggingFace."""
    print("Loading MMLU-Pro dataset...")
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    test_df = dataset["test"]
    val_df = dataset["validation"]
    test_df = preprocess(test_df)
    val_df = preprocess(val_df)
    return test_df, val_df


def preprocess(test_df):
    """Remove N/A options from dataset."""
    res_df = []
    for each in test_df:
        options = []
        for opt in each["options"]:
            if opt == "N/A":
                continue
            options.append(opt)
        each["options"] = options
        res_df.append(each)
    return res_df


def select_by_category(df, category):
    """Select all questions from a specific category."""
    return [item for item in df if item["category"] == category]


def format_cot_example(example, including_answer=True):
    """Format a single example in chain-of-thought style."""
    prompt = "Question:\n"
    question = example["question"]
    options = example["options"]
    prompt += question + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(options):
        prompt += "{}. {}\n".format(CHOICES[i], opt)
    
    if including_answer:
        cot_content = example.get("cot_content", "")
        if cot_content:
            cot_content = cot_content.replace("A: Let's think step by step.",
                                             "Answer: Let's think step by step.")
            prompt += cot_content + "\n\n"
        else:
            # If no cot_content, just add the answer
            prompt += f"Answer: The answer is ({CHOICES[example['answer_index']]}).\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    
    return prompt


def generate_cot_prompt(val_df, curr, k=5):
    """Generate chain-of-thought prompt with k few-shot examples."""
    # Initial prompt
    initial_prompt = (
        "The following are multiple choice questions (with answers) about {$}. "
        "Think step by step and then finish your answer with \"the answer is (X)\" "
        "where X is the correct letter choice.\n\n"
    )
    
    category = curr["category"]
    prompt = initial_prompt.replace("{$}", category)
    
    # Get examples from validation set for the same category
    category_examples = select_by_category(val_df, category)
    few_shot_examples = category_examples[:k]
    
    # Add few-shot examples
    for example in few_shot_examples:
        prompt += format_cot_example(example, including_answer=True)
    
    # Add current question
    prompt += format_cot_example(curr, including_answer=False)
    
    return prompt


def sample_questions(test_df, percentage, seed=SEED):
    """
    Randomly sample a percentage of questions from test set.
    
    Args:
        test_df: List of test questions
        percentage: Percentage of questions to sample (0-100)
        seed: Random seed for reproducibility
    
    Returns:
        List of sampled questions with indices
    """
    set_seed(seed)
    
    total_questions = len(test_df)
    num_samples = int(total_questions * percentage / 100.0)
    
    print(f"Total questions: {total_questions}")
    print(f"Sampling {percentage}% = {num_samples} questions")
    
    # Get random indices
    all_indices = list(range(total_questions))
    sampled_indices = random.sample(all_indices, num_samples)
    sampled_indices.sort()  # Sort for easier debugging
    
    # Get sampled questions
    sampled_questions = []
    for idx in sampled_indices:
        question = test_df[idx].copy()
        question['dataset_index'] = idx
        sampled_questions.append(question)
    
    return sampled_questions


def load_tokenizer(model_id):
    """Load tokenizer for the specified model."""
    print(f"Loading tokenizer for: {model_id}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id, 
        trust_remote_code=True,
        token=os.getenv("HF_TOKEN")
    )
    return tokenizer


def generate_prompts_for_questions(sampled_questions, val_df, k_shot=5, tokenizer=None):
    """
    Generate prompts for all sampled questions.
    
    Args:
        sampled_questions: List of sampled questions
        val_df: Validation set for few-shot examples
        k_shot: Number of few-shot examples
        tokenizer: Optional tokenizer to generate prompt_ids
    
    Returns:
        List of dicts with question_id, category, prompt, prompt_ids, and ground truth info
    """
    results = []
    
    print(f"Generating prompts for {len(sampled_questions)} questions...")
    for question in tqdm(sampled_questions):
        # Generate prompt
        prompt = generate_cot_prompt(val_df, question, k=k_shot)
        
        # Get ground truth answer
        answer_index = question["answer_index"]
        correct_answer = CHOICES[answer_index]
        
        # Create result entry
        result = {
            "question_id": question["question_id"],
            "category": question["category"],
            "prompt": prompt,
            "question": question["question"],
            "options": question["options"],
            "answer": correct_answer,
            "answer_index": answer_index
        }
        
        # Tokenize prompt if tokenizer is provided
        if tokenizer is not None:
            tokenized = tokenizer(prompt, return_tensors=None, truncation=True, max_length=4096)
            result["prompt_ids"] = tokenized["input_ids"]
        
        results.append(result)

    random.shuffle(results)
    
    return results


def save_to_json(data, output_file):
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nSaved {len(data)} questions to: {output_file}")


def print_statistics(data):
    """Print statistics about the sampled questions."""
    from collections import Counter
    
    categories = [item['category'] for item in data]
    category_counts = Counter(categories)
    
    print("\n" + "="*80)
    print("SAMPLE STATISTICS")
    print("="*80)
    print(f"\nTotal questions sampled: {len(data)}")
    print(f"\nQuestions per category:")
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Randomly sample MMLU-Pro questions and generate prompts"
    )
    parser.add_argument(
        "--percentage", 
        type=float, 
        required=True,
        help="Percentage of questions to sample (0-100)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="sample_questions.json",
        help="Output JSON file"
    )
    parser.add_argument(
        "--k_shot", 
        type=int, 
        default=5,
        help="Number of few-shot examples to include in prompts"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=SEED,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default=None,
        help="Model ID to use for tokenization. If provided, generates prompt_ids (list of token integers)"
    )
    
    args = parser.parse_args()
    
    # Validate percentage
    if args.percentage <= 0 or args.percentage > 100:
        print("Error: Percentage must be between 0 and 100")
        return
    
    # Load dataset
    test_df, val_df = load_mmlu_pro()
    
    # Sample questions
    sampled_questions = sample_questions(test_df, args.percentage, seed=args.seed)
    
    # Load tokenizer if model is specified
    tokenizer = None
    if args.model:
        tokenizer = load_tokenizer(args.model)
        print("Will generate prompt_ids (tokenized prompts)")
    else:
        print("No model specified, skipping tokenization (prompt_ids will not be generated)")
    
    # Generate prompts
    prompt_data = generate_prompts_for_questions(sampled_questions, val_df, k_shot=args.k_shot, tokenizer=tokenizer)
    
    # Print statistics
    print_statistics(prompt_data)
    
    # Save to JSON
    save_to_json(prompt_data, args.output)


if __name__ == "__main__":
    main()

