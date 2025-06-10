"""
evaluate_spatialbench.py

Minimal script to evaluate VLM models on SpatialBench benchmark.
Supports Cobra, OpenSpace Qwen, and SpaceThinker Qwen models.

Usage:
python scripts/evaluate_spatialbench.py --model_type cobra --model_path cobra+3b
python scripts/evaluate_spatialbench.py --model_type qwen --model_path remyxai/SpaceQwen2.5-VL-3B-Instruct
python scripts/evaluate_spatialbench.py --model_type qwen --model_path remyxai/SpaceThinker-Qwen2.5VL-3B
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def load_cobra_model(model_path: str, hf_token: str):
    """Load Cobra model."""
    from cobra import load
    
    model = load(model_path, hf_token=hf_token)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model.to(device, dtype=dtype)
    return model, None


def load_qwen_model(model_path: str):
    """Load Qwen model (OpenSpace or SpaceThinker)."""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    return model, processor


def generate_cobra_response(model, image: Image.Image, question: str, max_tokens: int = 512) -> str:
    """Generate response using Cobra model."""
    prompt_builder = model.get_prompt_builder()
    prompt_builder.add_turn(role="human", message=question)
    prompt_text = prompt_builder.get_prompt()
    
    response = model.generate(
        image,
        prompt_text,
        use_cache=True,
        do_sample=False,
        max_new_tokens=max_tokens,
    )
    return response


def generate_qwen_response(model, processor, image: Image.Image, question: str, max_tokens: int = 512) -> str:
    """Generate response using Qwen model."""
    # Prepare the conversation format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        }
    ]
    
    # Apply chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Process inputs
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt"
    ).to(model.device)
    
    # Generate response
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=0.2
        )
    
    # Decode response
    generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
    response = processor.decode(generated_ids, skip_special_tokens=True)
    return response


def extract_answer(response: str) -> str:
    """Extract short answer from model response."""
    # Take first sentence and clean it
    answer = response.split('.')[0].strip().lower()
    
    # Remove common prefixes
    prefixes = ["the answer is", "based on the image", "looking at the image", "the object is"]
    for prefix in prefixes:
        if answer.startswith(prefix):
            answer = answer[len(prefix):].strip(' ,.').lower()
    
    return answer


def evaluate_spatialbench(
    model_type: str,
    model_path: str,
    output_dir: str = "spatialbench_results",
    max_samples: Optional[int] = None,
    hf_token: Optional[str] = None
) -> Dict:
    """Main evaluation function."""
    
    # Load model
    print(f"Loading {model_type} model: {model_path}")
    if model_type == "cobra":
        if not hf_token:
            hf_token = Path(".hf_token").read_text().strip()
        model, tokenizer = load_cobra_model(model_path, hf_token)
    elif model_type == "qwen":
        model, processor = load_qwen_model(model_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Load SpatialBench dataset
    print("Loading SpatialBench dataset...")
    try:
        dataset = load_dataset("RussRobin/SpatialBench", split="test")
    except Exception as e:
        print(f"Error loading dataset. Make sure you have accepted the dataset terms: {e}")
        return {}
    
    # Limit samples if specified
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"Evaluating on {len(dataset)} samples")
    
    # Evaluate
    results = []
    correct = 0
    total = 0
    
    for idx, example in enumerate(tqdm(dataset, desc="Evaluating")):
        try:
            # Get question and image
            question = example["question"]
            image = example["image"]
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif hasattr(image, 'convert'):
                image = image.convert("RGB")
            
            # Generate response
            if model_type == "cobra":
                response = generate_cobra_response(model, image, question)
            elif model_type == "qwen":
                response = generate_qwen_response(model, processor, image, question)
            
            # Extract answer
            predicted_answer = extract_answer(response)
            
            # Get ground truth
            ground_truth = example.get("answer", "").lower().strip()
            
            # Check correctness
            is_correct = predicted_answer == ground_truth
            if is_correct:
                correct += 1
            total += 1
            
            # Store result
            results.append({
                "index": idx,
                "question": question,
                "predicted_answer": predicted_answer,
                "ground_truth": ground_truth,
                "full_response": response,
                "is_correct": is_correct
            })
            
        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            continue
    
    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0
    
    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    model_name = model_path.replace("/", "_").replace("+", "_")
    results_file = output_dir / f"{model_type}_{model_name}_spatialbench_results.json"
    
    final_results = {
        "model_type": model_type,
        "model_path": model_path,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results
    }
    
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nResults:")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Results saved to: {results_file}")
    
    return final_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM models on SpatialBench")
    parser.add_argument("--model_type", choices=["cobra", "qwen"], required=True,
                        help="Type of model to evaluate")
    parser.add_argument("--model_path", required=True,
                        help="Model path/identifier")
    parser.add_argument("--output_dir", default="spatialbench_results",
                        help="Output directory for results")
    parser.add_argument("--max_samples", type=int,
                        help="Maximum number of samples to evaluate")
    parser.add_argument("--hf_token", 
                        help="Hugging Face token (required for Cobra)")
    
    args = parser.parse_args()
    
    evaluate_spatialbench(
        model_type=args.model_type,
        model_path=args.model_path,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        hf_token=args.hf_token
    )


if __name__ == "__main__":
    main() 