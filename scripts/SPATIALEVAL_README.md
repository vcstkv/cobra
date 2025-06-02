# SpatialEval Evaluation for Cobra Model

This directory contains scripts to evaluate the Cobra vision-language model on the SpatialEval benchmark. SpatialEval tests spatial reasoning abilities across four tasks and three input modalities.

## Overview

**SpatialEval Benchmark:**
- **Tasks**: Spatial-Map, Maze-Nav, Spatial-Grid, Spatial-Real
- **Modalities**: TQA (text-only), VQA (vision-only), VTQA (vision-text)
- **Evaluation**: Exact match against ground truth answers
- **Dataset**: Available on Hugging Face as `MilaWang/SpatialEval`

## Setup

### Prerequisites

1. **Install dependencies:**
```bash
pip install datasets tqdm
```

2. **Hugging Face Token:**
Create a `.hf_token` file in the repository root with your Hugging Face token:
```bash
echo "your_hf_token_here" > .hf_token
```

3. **Directory Structure:**
The scripts will create the following directories:
```
outputs/              # Generated responses and results
eval_summary/         # Evaluation summaries and metrics
```

## Usage

### 1. Full Evaluation (Generate + Evaluate)

**Single Mode/Task:**
```bash
# Evaluate VQA mode on Spatial-Map task
python scripts/evaluate_spatialeval.py --mode vqa --task spatialmap

# Evaluate VTQA mode on all tasks
python scripts/evaluate_spatialeval.py --mode vtqa --task all

# Test with first 10 examples only
python scripts/evaluate_spatialeval.py --mode vqa --task spatialmap --first_k 10
```

**Batch Evaluation (Multiple modes/tasks):**
```bash
# Run all combinations of VQA/VTQA modes and all tasks
python scripts/run_spatialeval_batch.py

# Test with first 5 examples for quick validation
python scripts/run_spatialeval_batch.py --first_k 5

# Specify custom modes and tasks
python scripts/run_spatialeval_batch.py --modes vqa --tasks spatialmap mazenav
```

### 2. Evaluate Pre-generated Results

If you already have generated responses, use the standalone evaluation script:

```bash
# Evaluate existing results file
python scripts/evaluate_results.py \
    --results_file outputs/vqa_spatialmap_results.json \
    --mode vqa \
    --task spatialmap
```

## Configuration Options

### Model Configuration
- `--model_path`: Path to Cobra model (default: "cobra+3b")
- `--hf_token`: Path to HF token file (default: ".hf_token")

### Evaluation Configuration
- `--mode`: Input modality (tqa/vqa/vtqa)
- `--task`: Task to evaluate (all/spatialmap/mazenav/spatialgrid/spatialreal)
- `--first_k`: Limit to first k examples (useful for testing)

### Generation Parameters
- `--do_sample`: Whether to use sampling (default: False)
- `--temperature`: Sampling temperature (default: 0.2)
- `--max_new_tokens`: Maximum tokens to generate (default: 512)

### Output Configuration
- `--output_dir`: Directory for results (default: "outputs")
- `--eval_summary_dir`: Directory for summaries (default: "eval_summary")

## Expected Prompt Format

The SpatialEval questions expect responses in this format:
```
"First, provide a concise answer in one sentence. Then, elaborate on the reasoning behind your answer in a detailed, step-by-step explanation."
```

The evaluation extracts the first sentence as the answer for exact match comparison.

## Output Files

### 1. Detailed Results
**Location:** `outputs/{mode}_{task}_results.json`

**Format:**
```json
{
  "results": [
    {
      "index": 0,
      "question": "What is the spatial relationship...",
      "predicted_answer": "The object is to the left",
      "full_response": "The object is to the left. This can be determined by...",
      "oracle_answer": "left",
      "oracle_option": "to the left",
      "is_correct": true
    }
  ],
  "accuracy": 0.75,
  "correct_count": 3,
  "total_count": 4
}
```

### 2. Evaluation Summary
**Location:** `eval_summary/{mode}_{task}_summary.json`

**Format:**
```json
{
  "model_path": "cobra+3b",
  "mode": "vqa",
  "task": "spatialmap",
  "overall_accuracy": 0.65,
  "total_correct": 13,
  "total_examples": 20,
  "task_results": {
    "spatialmap": {"accuracy": 0.65, "correct": 13, "total": 20}
  },
  "config": {
    "do_sample": false,
    "temperature": 0.2,
    "max_new_tokens": 512,
    "first_k": null
  }
}
```

### 3. Batch Summary
**Location:** `eval_summary/batch_evaluation_summary.json`

Summarizes results across multiple mode/task combinations.

## Evaluation Details

### Answer Extraction
The scripts extract short answers from model responses by:
1. Taking the first sentence from the response
2. Removing common prefixes like "The answer is", "Based on the image"
3. Cleaning punctuation

### Exact Match Scoring
- Converts both predicted and ground truth to lowercase
- Removes punctuation (.,!?;:)
- Compares against both `oracle_answer` and `oracle_option` fields
- Returns True if either matches exactly

### Task Filtering
The dataset is automatically filtered by the `task` field to evaluate only the specified spatial reasoning task.

## Example Workflows

### Quick Test (5 examples per task)
```bash
# Test the system with a small subset
python scripts/run_spatialeval_batch.py --first_k 5
```

### Full VQA Evaluation
```bash
# Evaluate all VQA tasks
python scripts/evaluate_spatialeval.py --mode vqa --task all
```

### Comprehensive Evaluation
```bash
# Evaluate all mode/task combinations (takes longer)
python scripts/run_spatialeval_batch.py
```

### Re-evaluate Existing Results
```bash
# If you want to re-score without re-generating
python scripts/evaluate_results.py \
    --results_file outputs/vqa_all_results.json \
    --mode vqa \
    --task all
```

## Troubleshooting

### Common Issues

1. **Missing HF Token:**
   - Create `.hf_token` file with your Hugging Face token
   - Or set `HF_TOKEN` environment variable

2. **Model Loading Errors:**
   - Ensure you have the required dependencies for mamba-ssm (though it won't run)
   - Check that the model path is correct

3. **CUDA Memory Issues:**
   - Reduce `max_new_tokens`
   - Use `first_k` parameter to limit examples
   - Ensure GPU has sufficient memory

4. **Dataset Loading Issues:**
   - Check internet connection
   - Verify Hugging Face token has access
   - Try manual dataset download

### Performance Notes

- **VQA and VTQA modes** work best with vision-language models like Cobra
- **TQA mode** may not perform optimally since VLMs expect image inputs
- Use `first_k` parameter for quick testing and validation
- Full evaluation can take significant time depending on model size and GPU

## Results Interpretation

- **Accuracy**: Percentage of exact matches with ground truth
- **Per-task results**: Shows model performance on different spatial reasoning aspects
- **Detailed results**: Useful for error analysis and understanding failure modes

Good spatial reasoning performance typically shows:
- Higher accuracy on Spatial-Map (spatial relationships)
- Consistent performance across VQA and VTQA modes
- Reasonable accuracy on navigation tasks (Maze-Nav) 