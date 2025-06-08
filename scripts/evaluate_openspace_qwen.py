"""
evaluate_spaceqwen.py

Minimal script to evaluate SpaceQwen2.5-VL-3B-Instruct on SpatialEval benchmark.
"""

import json
from pathlib import Path
from typing import Dict, List

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Constants
MODEL_ID = "remyxai/SpaceQwen2.5-VL-3B-Instruct"
DATASET_ID = "MilaWang/SpatialEval"
TASKS = ["spatialmap", "mazenav", "spatialgrid", "spatialreal"]
MODES = ["vqa", "vtqa"]  # Excluding TQA since this is a VLM

def setup_model():
    """Initialize model and tokenizer."""
    print("Loading SpaceQwen model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    return model, tokenizer

def evaluate_example(model, tokenizer, example: Dict, mode: str) -> Dict:
    """Evaluate a single example."""
    try:
        # Prepare prompt
        question = example["text"]
        if "First, provide a concise answer" not in question:
            question += " First, provide a concise answer in one sentence. Then, elaborate on the reasoning behind your answer in a detailed, step-by-step explanation."

        # Handle image
        image = example["image"]
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif hasattr(image, 'convert'):
            image = image.convert("RGB")

        # Generate response
        inputs = tokenizer.build_multimodal_inputs(
            query=question,
            images=[image],
            return_tensors="pt"
        ).to(model.device)
        
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.2,
            top_p=0.9
        )
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Extract answer (first sentence)
        answer = response.split('.')[0].strip().lower()
        for prefix in ["the answer is", "based on the image", "looking at the image"]:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip(' ,.').lower()

        # Compare with ground truth
        gt_answer = example.get("oracle_answer", "").lower()
        gt_option = example.get("oracle_option", "").lower()
        is_correct = answer == gt_answer or answer == gt_option

        return {
            "question": question,
            "predicted": answer,
            "ground_truth": [gt_answer, gt_option],
            "full_response": response,
            "is_correct": is_correct
        }

    except Exception as e:
        print(f"Error processing example: {e}")
        return None

def evaluate_task(model, tokenizer, mode: str, task: str, first_k: int = None) -> Dict:
    """Evaluate model on a specific task and mode."""
    print(f"\nEvaluating {task} on {mode} mode...")
    
    # Load and filter dataset
    dataset = load_dataset(DATASET_ID, mode, split="test")
    task_data = dataset.filter(lambda x: task in x['id'])
    if first_k:
        task_data = task_data.select(range(min(first_k, len(task_data))))

    results = []
    correct = 0
    total = 0

    for example in tqdm(task_data, desc=f"{mode}-{task}"):
        result = evaluate_example(model, tokenizer, example, mode)
        if result:
            results.append(result)
            if result["is_correct"]:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0
    return {
        "task": task,
        "mode": mode,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results
    }

def main(output_dir: str = "spaceqwen_results", first_k: int = None):
    """Run evaluation on all tasks and modes."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    model, tokenizer = setup_model()
    all_results = {}

    for mode in MODES:
        mode_results = {}
        for task in TASKS:
            results = evaluate_task(model, tokenizer, mode, task, first_k)
            mode_results[task] = results
            
            # Save individual task results
            task_file = output_dir / f"{mode}_{task}_results.json"
            with open(task_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"{mode.upper()} {task}: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
        
        all_results[mode] = mode_results

    # Save overall results
    summary_file = output_dir / "spaceqwen_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\nEvaluation complete! Results saved to:", output_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="spaceqwen_results")
    parser.add_argument("--first_k", type=int, help="Evaluate only first k examples")
    args = parser.parse_args()
    
    main(args.output_dir, args.first_k) 