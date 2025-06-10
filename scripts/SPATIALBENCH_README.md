# SpatialBench Evaluation for VLM Models

This script evaluates Vision-Language Models on the SpatialBench benchmark, which tests spatial understanding capabilities across five task types: positional, existence, counting, reaching, and size comparison tasks.

## Supported Models

- **Cobra**: Your custom VLM model (`cobra+3b`)
- **OpenSpace Qwen**: `remyxai/SpaceQwen2.5-VL-3B-Instruct`
- **SpaceThinker Qwen**: `remyxai/SpaceThinker-Qwen2.5VL-3B`

## Prerequisites

### 1. Install Dependencies
```bash
pip install torch transformers datasets pillow tqdm
```

### 2. Dataset Access
You must accept the SpatialBench dataset terms on Hugging Face:
1. Visit: https://huggingface.co/datasets/RussRobin/SpatialBench
2. Log in to your Hugging Face account
3. Accept the dataset access conditions

### 3. Authentication (for Cobra model)
Create a `.hf_token` file in the repository root:
```bash
echo "your_hf_token_here" > .hf_token
```

## Usage

### Basic Commands

**Evaluate Cobra Model:**
```bash
python scripts/evaluate_spatialbench.py --model_type cobra --model_path cobra+3b
```

**Evaluate OpenSpace Qwen:**
```bash
python scripts/evaluate_spatialbench.py --model_type qwen --model_path remyxai/SpaceQwen2.5-VL-3B-Instruct
```

**Evaluate SpaceThinker Qwen:**
```bash
python scripts/evaluate_spatialbench.py --model_type qwen --model_path remyxai/SpaceThinker-Qwen2.5VL-3B
```

### Advanced Options

**Limit Sample Size (for testing):**
```bash
python scripts/evaluate_spatialbench.py \
    --model_type cobra \
    --model_path cobra+3b \
    --max_samples 100
```

**Custom Output Directory:**
```bash
python scripts/evaluate_spatialbench.py \
    --model_type qwen \
    --model_path remyxai/SpaceQwen2.5-VL-3B-Instruct \
    --output_dir my_results
```

**Custom HF Token (for Cobra):**
```bash
python scripts/evaluate_spatialbench.py \
    --model_type cobra \
    --model_path cobra+3b \
    --hf_token your_token_here
```

## Parameters

| Parameter | Required | Description | Default |
|-----------|----------|-------------|---------|
| `--model_type` | Yes | Model type: `cobra` or `qwen` | - |
| `--model_path` | Yes | Model identifier or path | - |
| `--output_dir` | No | Output directory for results | `spatialbench_results` |
| `--max_samples` | No | Limit number of samples (for testing) | All samples |
| `--hf_token` | No* | Hugging Face token (*required for Cobra) | Read from `.hf_token` |

## Output

### Console Output
```
Loading qwen model: remyxai/SpaceQwen2.5-VL-3B-Instruct
Loading SpatialBench dataset...
Evaluating on 956 samples
Evaluating: 100%|████████████| 956/956 [15:23<00:00,  1.03it/s]

Results:
Accuracy: 0.6544 (626/956)
Results saved to: spatialbench_results/qwen_remyxai_SpaceQwen2.5-VL-3B-Instruct_spatialbench_results.json
```

### Result Files
Results are saved as JSON files in the output directory:

**File naming pattern:**
- Cobra: `cobra_cobra_3b_spatialbench_results.json`
- OpenSpace: `qwen_remyxai_SpaceQwen2.5-VL-3B-Instruct_spatialbench_results.json`
- SpaceThinker: `qwen_remyxai_SpaceThinker-Qwen2.5VL-3B_spatialbench_results.json`

**JSON structure:**
```json
{
  "model_type": "qwen",
  "model_path": "remyxai/SpaceQwen2.5-VL-3B-Instruct",
  "accuracy": 0.6544,
  "correct": 626,
  "total": 956,
  "results": [
    {
      "index": 0,
      "question": "Is there a red object in the image?",
      "predicted_answer": "yes",
      "ground_truth": "yes",
      "full_response": "Yes, there is a red object in the image...",
      "is_correct": true
    }
  ]
}
```

## Batch Evaluation

To evaluate all three models sequentially:

```bash
# Evaluate all models
python scripts/evaluate_spatialbench.py --model_type cobra --model_path cobra+3b
python scripts/evaluate_spatialbench.py --model_type qwen --model_path remyxai/SpaceQwen2.5-VL-3B-Instruct  
python scripts/evaluate_spatialbench.py --model_type qwen --model_path remyxai/SpaceThinker-Qwen2.5VL-3B
```

Or create a simple bash script:
```bash
#!/bin/bash
# evaluate_all.sh

echo "Evaluating Cobra..."
python scripts/evaluate_spatialbench.py --model_type cobra --model_path cobra+3b

echo "Evaluating OpenSpace Qwen..."
python scripts/evaluate_spatialbench.py --model_type qwen --model_path remyxai/SpaceQwen2.5-VL-3B-Instruct

echo "Evaluating SpaceThinker Qwen..."
python scripts/evaluate_spatialbench.py --model_type qwen --model_path remyxai/SpaceThinker-Qwen2.5VL-3B

echo "All evaluations complete!"
```

## Evaluation Details

### Task Types
SpatialBench includes five types of spatial reasoning tasks:
- **Positional**: Spatial relationships between objects
- **Existence**: Whether objects exist in specific locations
- **Counting**: Number of objects in spatial regions
- **Reaching**: Reachability and path finding
- **Size Comparison**: Relative sizes of objects

### Answer Extraction
The script extracts answers by:
1. Taking the first sentence from the model response
2. Converting to lowercase
3. Removing common prefixes ("the answer is", "based on the image", etc.)
4. Stripping punctuation

### Scoring
- Uses exact match comparison between predicted and ground truth answers
- Case-insensitive matching
- Reports overall accuracy and detailed per-sample results

## Troubleshooting

### Common Issues

**Dataset Access Error:**
```
Error loading dataset. Make sure you have accepted the dataset terms
```
- Visit https://huggingface.co/datasets/RussRobin/SpatialBench and accept terms
- Ensure your HF token has the necessary permissions

**CUDA Memory Error:**
- Reduce `--max_samples` for testing
- Use CPU by setting `CUDA_VISIBLE_DEVICES=""`

**Model Loading Error (Cobra):**
- Ensure `.hf_token` file exists and contains valid token
- Check that the Cobra model path is correct

**Model Loading Error (Qwen):**
- Verify model identifier is correct
- Check internet connection for model download
- Ensure sufficient disk space for model weights

### Performance Tips

1. **Start Small**: Use `--max_samples 10` for initial testing
2. **Monitor Memory**: Large models may require significant GPU memory
3. **Check Progress**: The script shows a progress bar with completion estimates
4. **Save Intermediate**: Results are saved after completion, not during

## Expected Performance

Based on the SpatialBot paper, expected accuracy ranges:
- **Strong performance**: 60-80% on spatial reasoning tasks
- **Moderate performance**: 40-60% 
- **Baseline performance**: 20-40%

Results will vary based on model capabilities and spatial reasoning strengths.

## Comparison with SpatialBot

The original SpatialBot model achieves strong performance on SpatialBench. Your evaluation results can be compared against:
- **SpatialBot-3B**: Reference model from the paper
- **General VLMs**: Baseline performance on spatial tasks
- **Spatial-specialized models**: Other models designed for spatial reasoning

Use your results to understand the relative spatial reasoning capabilities of your three VLM models. 