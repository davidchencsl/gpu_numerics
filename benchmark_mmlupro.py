#!/usr/bin/env python3
"""
MMLU-Pro Evaluation Script using vLLM for fast inference.

Usage:
  # With pre-generated prompts (recommended):
  uv run benchmark_mmlupro.py --model MODEL_NAME --prompt_json sample_questions.json
  
  # Generate prompts on-the-fly:
  uv run benchmark_mmlupro.py --model MODEL_NAME --selected_subjects all
  
Features:
  - Fast batched inference with vLLM
  - Support for pre-generated prompts from JSON
  - Accuracy reporting by category and overall
  - Greedy decoding with temperature=0
  - Max output tokens: 128 (optimized for efficiency)
"""

import csv
import json
import argparse
import os
import pickle
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

choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
max_model_length = 4096
max_new_tokens = 1024

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

def load_mmlu_pro():
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    test_df, val_df = dataset["test"], dataset["validation"]
    test_df = preprocess(test_df)
    val_df = preprocess(val_df)
    return test_df, val_df


def load_model():
    llm = LLM(model=args.model,
                tensor_parallel_size=1,
                max_model_len=max_model_length,
                trust_remote_code=True,
                )
    sampling_params = SamplingParams(temperature=0, 
                                     max_tokens=max_new_tokens,
                                     seed=SEED,
                                     stop=["Question:"])
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, token=os.getenv("HF_TOKEN"))
    return (llm, sampling_params), tokenizer


def load_prompts_from_json(json_file):
    """Load pre-generated prompts from JSON file."""
    if not json_file or not os.path.exists(json_file):
        return None
    
    logging.info(f"Loading prompts from: {json_file}")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    logging.info(f"Loaded {len(data)} pre-generated prompts")
    return data


def preprocess(test_df):
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


def args_generate_path(input_args):
    scoring_method = "CoT"
    model_name = input_args.model.split("/")[-1]
    subjects = args.selected_subjects.replace(",", "-").replace(" ", "_")
    return [model_name, scoring_method, subjects]


def select_by_category(df, subject):
    res = []
    for each in df:
        if each["category"] == subject:
            res.append(each)
    return res


def format_cot_example(example, including_answer=True):
    prompt = "Question:\n"
    question = example["question"]
    options = example["options"]
    prompt += question + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(options):
        prompt += "{}. {}\n".format(choices[i], opt)
    if including_answer:
        cot_content = example["cot_content"].replace("A: Let's think step by step.",
                                                     "Answer: Let's think step by step.")
        prompt += cot_content + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    return prompt


def generate_cot_prompt(val_df, curr, k):
    prompt = ""
    with open(f"cot_prompt_lib/initial_prompt.txt", "r") as fi:
        for line in fi.readlines():
            prompt += line
    subject = curr["category"]
    val_df = select_by_category(val_df, subject)
    val_df = val_df[: k]
    prompt = prompt.replace("{$}", subject) + "\n"
    for example in val_df:
        prompt += format_cot_example(example, including_answer=True)
    prompt += format_cot_example(curr, including_answer=False)
    return prompt


def extract_answer(text):
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        print("1st answer extract failed\n" + text)
        return extract_again(text)


def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)


def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None


def batch_inference(llm, sampling_params, inference_batch):
    start = time.time()
    outputs = llm.generate(inference_batch, sampling_params)
    logging.info(str(len(inference_batch)) + "size batch costing time: " + str(time.time() - start))
    response_batch = []
    pred_batch = []
    token_info_batch = []
    
    for output in outputs:
        generated_text = output.outputs[0].text
        response_batch.append(generated_text)
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


def save_res(res, output_path):
    accu, corr, wrong = 0.0, 0.0, 0.0
    
    # Save results to JSON with proper formatting
    with open(output_path, "w") as fo:
        json.dump(res, fo, indent=2)
    
    # Calculate token statistics
    total_prompt_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    
    for each in res:
        # Accumulate token counts
        if "prompt_tokens" in each:
            total_prompt_tokens += each["prompt_tokens"]
            total_output_tokens += each["output_tokens"]
            total_tokens += each["total_tokens"]
        
        # Calculate accuracy
        if not each["pred"]:
            x = random.randint(0, len(each["options"]) - 1)
            if x == each["answer_index"]:
                corr += 1
                # print("random hit.")
            else:
                wrong += 1
        elif each["pred"] == each["answer"]:
            corr += 1
        else:
            wrong += 1
    
    if corr + wrong == 0:
        return 0.0, 0.0, 0.0
    
    accu = corr / (corr + wrong)
    
    # Log token statistics
    if total_tokens > 0:
        logging.info(f"Token usage - Prompt: {total_prompt_tokens}, Output: {total_output_tokens}, Total: {total_tokens}")
    
    return accu, corr, wrong


@torch.no_grad()
def eval_cot(subject, model, tokenizer, val_df, test_df, output_path):
    llm, sampling_params = model
    global choices
    logging.info("evaluating " + subject)
    inference_batches = []

    for i in tqdm(range(len(test_df))):
        k = args.ntrain
        curr = test_df[i]
        prompt_length_ok = False
        prompt = None
        while not prompt_length_ok:
            prompt = generate_cot_prompt(val_df, curr, k)
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {key: value.cuda() for key, value in inputs.items()}
            length = len(inputs["input_ids"][0])
            if length < max_model_length - max_new_tokens:
                prompt_length_ok = True
            k -= 1
        inference_batches.append(prompt)

    pred_batch, response_batch, token_info_batch = batch_inference(llm, sampling_params, inference_batches)

    res = []
    for j, curr in enumerate(test_df):
        curr["pred"] = pred_batch[j]
        curr["model_outputs"] = response_batch[j]
        curr["prompt_tokens"] = token_info_batch[j]["prompt_tokens"]
        curr["output_tokens"] = token_info_batch[j]["output_tokens"]
        curr["total_tokens"] = token_info_batch[j]["total_tokens"]
        res.append(curr)
    accu, corr, wrong = save_res(res, output_path)
    logging.info("this batch accu is: {}, corr: {}, wrong: {}\n".format(str(accu), str(corr), str(wrong)))

    accu, corr, wrong = save_res(res, output_path)
    return accu, corr, wrong


@torch.no_grad()
def eval_from_prompts(prompt_data, model, output_path):
    """Evaluate using pre-generated prompts from JSON."""
    llm, sampling_params = model
    global choices
    
    logging.info(f"Evaluating {len(prompt_data)} questions from prompts...")
    
    # Extract prompts
    inference_batches = [item["prompt"] for item in prompt_data]
    
    # Run batched inference
    pred_batch, response_batch, token_info_batch = batch_inference(llm, sampling_params, inference_batches)
    
    # Compile results
    res = []
    for j, item in enumerate(prompt_data):
        result = {
            "question_id": item["question_id"],
            "category": item["category"],
            "question": item.get("question", ""),
            "options": item.get("options", []),
            "answer": item["answer"],
            "answer_index": item.get("answer_index", -1),
            "pred": pred_batch[j],
            "model_outputs": response_batch[j],
            "prompt_tokens": token_info_batch[j]["prompt_tokens"],
            "output_tokens": token_info_batch[j]["output_tokens"],
            "total_tokens": token_info_batch[j]["total_tokens"]
        }
        res.append(result)
    
    accu, corr, wrong = save_res(res, output_path)
    logging.info("Accuracy: {:.4f}, Correct: {}, Wrong: {}\n".format(accu, corr, wrong))
    
    return accu, corr, wrong


def main():
    model, tokenizer = load_model()
    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir)

    # Check if using pre-generated prompts from JSON
    if args.prompt_json:
        prompt_data = load_prompts_from_json(args.prompt_json)
        if prompt_data is None:
            logging.error(f"Error: Could not load prompt JSON file: {args.prompt_json}")
            return
        
        # Group prompts by category
        from collections import defaultdict
        prompts_by_category = defaultdict(list)
        for item in prompt_data:
            prompts_by_category[item["category"]].append(item)
        
        sta_dict = {}
        selected_subjects = sorted(prompts_by_category.keys())
        
        logging.info("Categories in prompt file:\n" + "\n".join(selected_subjects))
        print("Categories in prompt file:\n" + "\n".join(selected_subjects))
        
        with open(os.path.join(summary_path), 'a') as f:
            f.write("\n------category level sta------\n")
        
        for subject in selected_subjects:
            if subject not in sta_dict:
                sta_dict[subject] = {"corr": 0.0, "wrong": 0.0, "accu": 0.0}
            
            subject_prompts = prompts_by_category[subject]
            output_path = os.path.join(save_result_dir, "{}.json".format(subject))
            acc, corr_count, wrong_count = eval_from_prompts(subject_prompts, model, output_path)
            
            sta_dict[subject]["corr"] = corr_count
            sta_dict[subject]["wrong"] = wrong_count
            sta_dict[subject]["accu"] = acc
            
            with open(os.path.join(summary_path), 'a') as f:
                f.write("Average accuracy {:.4f} - {}\n".format(sta_dict[subject]["accu"], subject))
    
    else:
        # Original behavior: load full dataset and generate prompts
        full_test_df, full_val_df = load_mmlu_pro()
        all_subjects = []
        for each in full_test_df:
            if each["category"] not in all_subjects:
                all_subjects.append(each["category"])
        if args.selected_subjects == "all":
            selected_subjects = all_subjects
        else:
            selected_subjects = []
            args_selected = args.selected_subjects.split(",")
            for sub in all_subjects:
                for each in args_selected:
                    if each.replace(" ", "_") in sub.replace(" ", "_"):
                        selected_subjects.append(sub)
        logging.info("selected subjects:\n" + "\n".join(selected_subjects))
        print("selected subjects:\n" + "\n".join(selected_subjects))
        sta_dict = {}
        selected_subjects = sorted(selected_subjects)
        with open(os.path.join(summary_path), 'a') as f:
            f.write("\n------category level sta------\n")
        for subject in selected_subjects:
            if subject not in sta_dict:
                sta_dict[subject] = {"corr": 0.0, "wrong": 0.0, "accu": 0.0}
            test_df = select_by_category(full_test_df, subject)
            val_df = select_by_category(full_val_df, subject)
            output_path = os.path.join(save_result_dir, "{}.json".format(subject))
            acc, corr_count, wrong_count = eval_cot(subject, model, tokenizer, val_df, test_df, output_path)
            sta_dict[subject]["corr"] = corr_count
            sta_dict[subject]["wrong"] = wrong_count
            sta_dict[subject]["accu"] = acc
            with open(os.path.join(summary_path), 'a') as f:
                f.write("Average accuracy {:.4f} - {}\n".format(sta_dict[subject]["accu"], subject))
    
    # Compute totals
    total_corr, total_wrong = 0.0, 0.0
    for k, v in sta_dict.items():
        total_corr += v["corr"]
        total_wrong += v["wrong"]
    total_accu = total_corr / (total_corr + total_wrong + 0.000001)
    sta_dict["total"] = {"corr": total_corr, "wrong": total_wrong, "accu": total_accu}

    with open(os.path.join(summary_path), 'a') as f:
        f.write("\n------average acc sta------\n")
        weighted_acc = total_accu
        f.write("Average accuracy: {:.4f}\n".format(weighted_acc))
    with open(global_record_file, 'a', newline='') as file:
        writer = csv.writer(file)
        record = args_generate_path(args) + [time_str, weighted_acc]
        writer.writerow(record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--selected_subjects", "-sub", type=str, default="all")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--global_record_file", "-grf", type=str,
                        default="eval_record_collection.csv")
    parser.add_argument("--gpu_util", "-gu", type=str, default="0.8")
    parser.add_argument("--model", "-m", type=str, default="Qwen/Qwen3-30B-A3B-FP8")
    parser.add_argument("--prompt_json", type=str, default=None,
                        help="Path to JSON file with pre-generated prompts (from generate_question_sample.py)")

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    global_record_file = args.global_record_file
    save_result_dir = os.path.join(
        args.save_dir, "/".join(args_generate_path(args))
    )
    file_prefix = "-".join(args_generate_path(args))
    timestamp = time.time()
    time_str = time.strftime('%m-%d_%H-%M', time.localtime(timestamp))
    file_name = f"{file_prefix}_{time_str}_summary.txt"
    summary_path = os.path.join(args.save_dir, "summary", file_name)
    os.makedirs(os.path.join(args.save_dir, "summary"), exist_ok=True)
    os.makedirs(save_result_dir, exist_ok=True)
    save_log_dir = os.path.join(args.save_dir, "log")
    os.makedirs(save_log_dir, exist_ok=True)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(os.path.join(save_log_dir,
                                                                   file_name.replace("_summary.txt",
                                                                                     "_logfile.log"))),
                                  logging.StreamHandler(sys.stdout)])

    main()


