# SpatialBot Evaluation on SpatialEval Benchmark

This guide explains how to evaluate your SpatialBot model using the SpatialEval benchmark.

## Prerequisites

Install the required dependencies:

```bash
pip install torch transformers accelerate datasets pillow numpy tqdm
```

## Quick Start

### Basic Evaluation

Run a basic evaluation on the VQA mode (vision-only) for all tasks:

```bash
python spatialbot_eval.py
```

### Evaluation Options

The script supports several evaluation modes and tasks:

#### Modes:
- `tqa`: Text-only questions (no images)
- `vqa`: Vision-only questions (images without text context)
- `vtqa`: Vision-text questions (images with text context)

#### Tasks:
- `all`: All four tasks (default)
- `spatialmap`: Spatial relationships in maps
- `mazenav`: Navigation through mazes
- `spatialgrid`: Spatial reasoning in grids
- `spatialreal`: Real-world spatial understanding

### Example Commands

```bash
# Evaluate on text-only mode for all tasks
python spatialbot_eval.py --mode tqa --task all

# Evaluate on vision-text mode for spatial-map task only
python spatialbot_eval.py --mode vtqa --task spatialmap

# Evaluate on first 100 samples only (for quick testing)
python spatialbot_eval.py --max_samples 100

# Use a local model path
python spatialbot_eval.py --model_name /path/to/your/spatialbot-model

# Save results to a specific file
python spatialbot_eval.py --output_file my_evaluation_results.json
```

### Full Command Options

```bash
python spatialbot_eval.py \
    --model_name RussRobin/SpatialBot-3B \
    --mode vqa \
    --task all \
    --device cuda \
    --max_samples 1000 \
    --output_file spatialbot_results.json
```

## Parameters

- `--model_name`: SpatialBot model name or local path (default: "RussRobin/SpatialBot-3B")
- `--mode`: Evaluation mode - tqa, vqa, or vtqa (default: "vqa")
- `--task`: Task to evaluate - all, spatialmap, mazenav, spatialgrid, or spatialreal (default: "all")
- `--device`: Device to use - cuda or cpu (default: "cuda")
- `--max_samples`: Maximum number of samples to evaluate (optional, for quick testing)
- `--output_file`: Output JSON file for results (default: "spatialbot_eval_results.json")

## Output

The script will:

1. Load the SpatialBot model
2. Load the SpatialEval dataset
3. Run inference on each sample
4. Calculate exact match accuracy
5. Save detailed results to a JSON file

### Output Format

The results JSON file contains:
- `config`: Configuration used for evaluation
- `accuracy`: Overall accuracy percentage
- `results`: Detailed results for each sample including:
  - Sample ID and task
  - Question text
  - Predicted answer
  - Full model response
  - Ground truth answer
  - Whether the prediction was correct

### Example Output

```
=== Evaluation Results ===
Mode: vqa
Task: all
Total samples: 1000
Correct: 650
Accuracy: 65.00%

Results saved to: spatialbot_eval_results.json
```

## Tips for Better Evaluation

1. **Start Small**: Use `--max_samples 10` for initial testing
2. **Task-Specific**: Evaluate individual tasks to understand performance patterns
3. **Mode Comparison**: Compare performance across tqa, vqa, and vtqa modes
4. **Local Models**: Use local model paths for faster loading if you have the model downloaded

## Understanding the Results

The evaluation uses exact match accuracy, which means:
- Predictions must exactly match the ground truth (case-insensitive)
- Only the first sentence of the model's response is used as the answer
- The model is prompted to provide a concise answer first, then elaborate

## Troubleshooting

### Common Issues:

1. **CUDA out of memory**: Reduce `max_samples` or use `--device cpu`
2. **Model loading errors**: Ensure you have the correct model name/path
3. **Dataset loading issues**: Check your internet connection for Hugging Face datasets

### Performance Notes:

- GPU evaluation is significantly faster than CPU
- Full evaluation can take several hours depending on your hardware
- Progress is shown every 50 samples

## SpatialEval Benchmark Overview

SpatialEval tests spatial reasoning across four dimensions:
1. **Spatial-Map**: Understanding relationships between objects in maps
2. **Maze-Nav**: Navigation through complex environments  
3. **Spatial-Grid**: Spatial reasoning within structured grids
4. **Spatial-Real**: Real-world spatial understanding

Each task tests different aspects of spatial intelligence, making it a comprehensive benchmark for evaluating spatial reasoning capabilities. 