"""
evaluate_spatialeval.py

Script to evaluate Cobra model on SpatialEval benchmark.
Supports TQA (text-only), VQA (vision-only), and VTQA (vision-text) modalities
across four tasks: Spatial-Map, Maze-Nav, Spatial-Grid, and Spatial-Real.

Usage:
python scripts/evaluate_spatialeval.py --model_path cobra+3b --mode vqa --task spatialmap --output_dir outputs
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import draccus
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from cobra import load
from cobra.overwatch import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


@dataclass
class EvaluationConfig:
    # Model Configuration
    model_path: Union[str, Path] = "cobra+3b"
    hf_token: Union[str, Path] = Path(".hf_token")
    
    # Evaluation Configuration
    mode: str = "vqa"  # tqa, vqa, vtqa
    task: str = "all"  # all, spatialmap, mazenav, spatialgrid, spatialreal
    dataset_id: str = "MilaWang/SpatialEval"
    
    # Output Configuration
    output_dir: Union[str, Path] = "outputs"
    eval_summary_dir: Union[str, Path] = "eval_summary"
    
    # Generation Parameters
    do_sample: bool = False
    temperature: float = 0.2
    max_new_tokens: int = 512
    
    # Evaluation Parameters
    first_k: Optional[int] = None  # Evaluate only first k examples


def extract_short_answer(response: str) -> str:
    """
    Extract the concise answer from the model response.
    Expected format: "First, provide a concise answer in one sentence. Then, elaborate..."
    """
    # Try to find the first sentence/short answer
    # Look for patterns like "The answer is X" or direct answers
    response = response.strip()
    
    # Split by sentences and take the first one as the short answer
    sentences = re.split(r'[.!?]', response)
    if sentences:
        first_sentence = sentences[0].strip()
        
        # Clean up common prefixes
        prefixes_to_remove = [
            "the answer is",
            "the correct answer is", 
            "based on the image",
            "looking at the image",
            "from the image",
            "in the image"
        ]
        
        lower_sentence = first_sentence.lower()
        for prefix in prefixes_to_remove:
            if lower_sentence.startswith(prefix):
                first_sentence = first_sentence[len(prefix):].strip()
                if first_sentence.startswith(','):
                    first_sentence = first_sentence[1:].strip()
                break
        
        return first_sentence
    
    return response


def exact_match(predicted: str, ground_truth: str) -> bool:
    """
    Compute exact match between predicted and ground truth answers.
    """
    # Normalize both strings
    pred_clean = predicted.lower().strip()
    gt_clean = ground_truth.lower().strip()
    
    # Remove common punctuation
    for char in ['.', ',', '!', '?', ';', ':']:
        pred_clean = pred_clean.replace(char, '')
        gt_clean = gt_clean.replace(char, '')
    
    return pred_clean == gt_clean


def evaluate_model_on_dataset(model, dataset, config: EvaluationConfig) -> Dict:
    """
    Evaluate the model on a dataset and return results.
    """
    device = next(model.parameters()).device
    results = []
    correct_count = 0
    total_count = 0
    
    # Get prompt builder for the model
    prompt_builder = model.get_prompt_builder()
    
    # Limit dataset if first_k is specified
    if config.first_k:
        dataset = dataset.select(range(min(config.first_k, len(dataset))))
    
    overwatch.info(f"Evaluating on {len(dataset)} examples")
    
    for idx, example in enumerate(tqdm(dataset, desc="Evaluating")):
        try:
            # Get text prompt
            text_prompt = example["text"]
            
            # Add reasoning instruction if not already present
            if "First, provide a concise answer" not in text_prompt:
                text_prompt += " First, provide a concise answer in one sentence. Then, elaborate on the reasoning behind your answer in a detailed, step-by-step explanation."
            
            # Handle different modalities
            image = None
            if config.mode in ["vqa", "vtqa"]:
                image = example["image"]
                if isinstance(image, str):
                    # If image is a path, load it
                    image = Image.open(image).convert("RGB")
                elif hasattr(image, 'convert'):
                    image = image.convert("RGB")
            
            # For TQA mode, we don't use images
            if config.mode == "tqa":
                image = None
            
            # Build prompt
            prompt_builder_instance = model.get_prompt_builder()
            prompt_builder_instance.add_turn(role="human", message=text_prompt)
            prompt_text = prompt_builder_instance.get_prompt()
            
            # Generate response
            if image is not None:
                generated_text = model.generate(
                    image,
                    prompt_text,
                    use_cache=True,
                    do_sample=config.do_sample,
                    temperature=config.temperature,
                    max_new_tokens=config.max_new_tokens,
                )
            else:
                # For text-only mode, we might need to handle differently
                # For now, use a dummy image if the model requires it
                dummy_image = Image.new('RGB', (224, 224), color='white')
                generated_text = model.generate(
                    dummy_image,
                    prompt_text,
                    use_cache=True,
                    do_sample=config.do_sample,
                    temperature=config.temperature,
                    max_new_tokens=config.max_new_tokens,
                )
            
            # Extract short answer
            predicted_answer = extract_short_answer(generated_text)
            
            # Get ground truth answers
            oracle_answer = example.get("oracle_answer", "")
            oracle_option = example.get("oracle_option", "")
            
            # Check exact match against oracle_answer and oracle_option
            is_correct = (exact_match(predicted_answer, oracle_answer) or 
                         exact_match(predicted_answer, oracle_option))
            
            if is_correct:
                correct_count += 1
            total_count += 1
            
            # Store result
            result = {
                "index": idx,
                "question": text_prompt,
                "predicted_answer": predicted_answer,
                "full_response": generated_text,
                "oracle_answer": oracle_answer,
                "oracle_option": oracle_option,
                "is_correct": is_correct
            }
            results.append(result)
            
        except Exception as e:
            overwatch.error(f"Error processing example {idx}: {e}")
            continue
    
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    
    return {
        "results": results,
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_count": total_count
    }


@draccus.wrap()
def evaluate_spatialeval(cfg: EvaluationConfig) -> None:
    """
    Main evaluation function.
    """
    overwatch.info(f"Starting SpatialEval evaluation with model: {cfg.model_path}")
    overwatch.info(f"Mode: {cfg.mode}, Task: {cfg.task}")
    
    # Setup device and dtype
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    # Load model
    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    overwatch.info("Loading model...")
    
    try:
        vlm = load(cfg.model_path, hf_token=hf_token)
        vlm.to(device, dtype=dtype)
        overwatch.info("Model loaded successfully")
    except Exception as e:
        overwatch.error(f"Failed to load model: {e}")
        return
    
    # Create output directories
    output_dir = Path(cfg.output_dir)
    eval_summary_dir = Path(cfg.eval_summary_dir)
    output_dir.mkdir(exist_ok=True)
    eval_summary_dir.mkdir(exist_ok=True)
    
    # Define tasks to evaluate
    if cfg.task == "all":
        tasks = ["spatialmap", "mazenav", "spatialgrid", "spatialreal"]
    else:
        tasks = [cfg.task]
    
    # Load dataset
    overwatch.info(f"Loading dataset: {cfg.dataset_id}")
    try:
        dataset = load_dataset(cfg.dataset_id, cfg.mode, split="test")
        overwatch.info(f"Dataset loaded with {len(dataset)} examples")
    except Exception as e:
        overwatch.error(f"Failed to load dataset: {e}")
        return
    
    # Evaluate on each task
    all_results = {}
    
    for task in tasks:
        overwatch.info(f"Evaluating on task: {task}")
        
        # Filter dataset by task
        task_dataset = dataset.filter(lambda x: x["task"] == task)
        overwatch.info(f"Task {task} has {len(task_dataset)} examples")
        
        if len(task_dataset) == 0:
            overwatch.warning(f"No examples found for task {task}")
            continue
        
        # Evaluate model
        task_results = evaluate_model_on_dataset(vlm, task_dataset, cfg)
        all_results[task] = task_results
        
        # Save task-specific results
        task_output_file = output_dir / f"{cfg.mode}_{task}_results.json"
        with open(task_output_file, 'w') as f:
            json.dump(task_results, f, indent=2)
        
        overwatch.info(f"Task {task} accuracy: {task_results['accuracy']:.4f} "
                      f"({task_results['correct_count']}/{task_results['total_count']})")
    
    # Compute overall results
    total_correct = sum(results['correct_count'] for results in all_results.values())
    total_examples = sum(results['total_count'] for results in all_results.values())
    overall_accuracy = total_correct / total_examples if total_examples > 0 else 0.0
    
    # Create evaluation summary
    summary = {
        "model_path": str(cfg.model_path),
        "mode": cfg.mode,
        "task": cfg.task,
        "overall_accuracy": overall_accuracy,
        "total_correct": total_correct,
        "total_examples": total_examples,
        "task_results": {task: {"accuracy": results["accuracy"], 
                               "correct": results["correct_count"],
                               "total": results["total_count"]} 
                        for task, results in all_results.items()},
        "config": {
            "do_sample": cfg.do_sample,
            "temperature": cfg.temperature,
            "max_new_tokens": cfg.max_new_tokens,
            "first_k": cfg.first_k
        }
    }
    
    # Save evaluation summary
    summary_file = eval_summary_dir / f"{cfg.mode}_{cfg.task}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print results
    overwatch.info("=" * 50)
    overwatch.info("EVALUATION RESULTS")
    overwatch.info("=" * 50)
    overwatch.info(f"Overall Accuracy: {overall_accuracy:.4f} ({total_correct}/{total_examples})")
    overwatch.info("Task-specific results:")
    for task, results in all_results.items():
        accuracy = results['accuracy']
        correct = results['correct_count']
        total = results['total_count']
        overwatch.info(f"  {task}: {accuracy:.4f} ({correct}/{total})")
    
    overwatch.info(f"Results saved to: {output_dir}")
    overwatch.info(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    evaluate_spatialeval() 