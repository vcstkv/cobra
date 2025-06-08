#!/usr/bin/env python3
"""
SpatialBot Evaluation on SpatialEval Benchmark

A minimal script to evaluate SpatialBot on the SpatialEval benchmark.
"""

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from PIL import Image
import argparse
import json
import os
import re
from tqdm import tqdm
import warnings
import numpy as np

# Disable warnings
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
warnings.filterwarnings('ignore')

def load_spatialbot_model(model_name, device):
    """Load SpatialBot model and tokenizer"""
    print(f"Loading SpatialBot model: {model_name}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    return model, tokenizer

def process_depth_image(image):
    """Process depth image for SpatialBot (convert single channel to RGB)"""
    if image.mode == 'L':  # Single channel depth image
        img = np.array(image)
        height, width = img.shape
        three_channel_array = np.zeros((height, width, 3), dtype=np.uint8)
        three_channel_array[:, :, 0] = (img // 1024) * 4
        three_channel_array[:, :, 1] = (img // 32) * 8
        three_channel_array[:, :, 2] = (img % 32) * 8
        return Image.fromarray(three_channel_array, 'RGB')
    return image

def prepare_spatialbot_input(prompt, images, tokenizer, model, device):
    """Prepare input for SpatialBot model"""
    # Format prompt for SpatialBot
    if len(images) == 1:
        text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image 1>\n{prompt} ASSISTANT:"
        text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image 1>\n')]
        input_ids = torch.tensor(text_chunks[0] + [-201] + text_chunks[1][1:], dtype=torch.long).unsqueeze(0).to(device)
    elif len(images) == 2:
        text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image 1>\n<image 2>\n{prompt} ASSISTANT:"
        text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image 1>\n<image 2>\n')]
        input_ids = torch.tensor(text_chunks[0] + [-201] + [-202] + text_chunks[1][1:], dtype=torch.long).unsqueeze(0).to(device)
    else:
        # Text-only mode
        text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {prompt} ASSISTANT:"
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        return input_ids, None
    
    # Process images
    processed_images = []
    for img in images:
        if img is not None:
            processed_images.append(process_depth_image(img))
    
    image_tensor = model.process_images(processed_images, model.config).to(dtype=model.dtype, device=device)
    
    return input_ids, image_tensor

def extract_answer(response):
    """Extract the first sentence (answer) from model response"""
    # Split by sentence endings and take the first sentence
    sentences = re.split(r'[.!?]+', response.strip())
    if sentences:
        return sentences[0].strip()
    return response.strip()

def evaluate_exact_match(predicted, ground_truth):
    """Simple exact match evaluation (case-insensitive)"""
    predicted = predicted.lower().strip()
    ground_truth = ground_truth.lower().strip()
    return predicted == ground_truth

def run_evaluation(model_name, mode, task, device, max_samples=None):
    """Run evaluation on SpatialEval"""
    
    # Load model
    model, tokenizer = load_spatialbot_model(model_name, device)
    
    # Load dataset
    print(f"Loading SpatialEval dataset: {mode}")
    dataset = load_dataset("MilaWang/SpatialEval", mode, split="test")
    
    # Filter by task if specified
    if task != "all":
        dataset = dataset.filter(lambda x: x.get('task', '').lower() == task.lower())
    
    # Limit samples if specified
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"Evaluating on {len(dataset)} samples")
    
    results = []
    correct = 0
    total = 0
    
    for i, sample in enumerate(tqdm(dataset, desc="Evaluating")):
        try:
            # Prepare prompt
            prompt_text = sample['text']
            prompt = f"{prompt_text} First, provide a concise answer in one sentence. Then, elaborate on the reasoning behind your answer in a detailed, step-by-step explanation."
            
            # Prepare images based on mode
            images = []
            if mode in ['vqa', 'vtqa'] and 'image' in sample:
                images.append(sample['image'])
            
            # Prepare input
            input_ids, image_tensor = prepare_spatialbot_input(prompt, images, tokenizer, model, device)
            
            # Generate response
            with torch.no_grad():
                if image_tensor is not None:
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor,
                        max_new_tokens=100,
                        use_cache=True,
                        repetition_penalty=1.0
                    )[0]
                else:
                    output_ids = model.generate(
                        input_ids,
                        max_new_tokens=100,
                        use_cache=True,
                        repetition_penalty=1.0
                    )[0]
            
            # Decode response
            response = tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
            
            # Extract answer (first sentence)
            predicted_answer = extract_answer(response)
            
            # Get ground truth
            ground_truth = sample.get('oracle_answer', '')
            
            # Evaluate
            is_correct = evaluate_exact_match(predicted_answer, ground_truth)
            
            results.append({
                'sample_id': i,
                'task': sample.get('task', ''),
                'question': sample['text'],
                'predicted_answer': predicted_answer,
                'full_response': response,
                'ground_truth': ground_truth,
                'correct': is_correct
            })
            
            if is_correct:
                correct += 1
            total += 1
            
            # Print progress every 50 samples
            if (i + 1) % 50 == 0:
                current_acc = correct / total * 100
                print(f"Progress: {i+1}/{len(dataset)}, Current Accuracy: {current_acc:.2f}%")
                
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
    
    # Calculate final accuracy
    accuracy = correct / total * 100 if total > 0 else 0
    
    print(f"\n=== Evaluation Results ===")
    print(f"Mode: {mode}")
    print(f"Task: {task}")
    print(f"Total samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    return results, accuracy

def main():
    parser = argparse.ArgumentParser(description="Evaluate SpatialBot on SpatialEval")
    parser.add_argument("--model_name", default="RussRobin/SpatialBot-3B", help="SpatialBot model name or path")
    parser.add_argument("--mode", choices=["tqa", "vqa", "vtqa"], default="vqa", help="Evaluation mode")
    parser.add_argument("--task", choices=["all", "spatialmap", "mazenav", "spatialgrid", "spatialreal"], default="all", help="Task to evaluate")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--max_samples", type=int, help="Maximum number of samples to evaluate")
    parser.add_argument("--output_file", default="spatialbot_eval_results.json", help="Output file for results")
    
    args = parser.parse_args()
    
    # Run evaluation
    results, accuracy = run_evaluation(
        model_name=args.model_name,
        mode=args.mode,
        task=args.task,
        device=args.device,
        max_samples=args.max_samples
    )
    
    # Save results
    output_data = {
        'config': vars(args),
        'accuracy': accuracy,
        'results': results
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {args.output_file}")

if __name__ == "__main__":
    main() 