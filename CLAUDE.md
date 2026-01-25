# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the codebase for the paper "Instructed to Bias: Instruction-Tuned Language Models Exhibit Emergent Cognitive Bias" (TACL). It investigates three cognitive biases in LLMs: the decoy effect, certainty effect, and belief bias.

## Common Commands

### Setup
```bash
pip install -r requirements.txt
```

### Generate Data
```bash
# BIAS_NAME: decoy, certainty, or false_belief
python Data_generation/generate_samples.py --bias_name $BIAS_NAME
```

### Run Predictions
```bash
# Example with T5 models
python run_predict.py --bias_name $BIAS_NAME --all_models t5-v1_1-small,flan-t5-small

# For OpenAI models, create .env file with: OPENAI_API_KEY=YOUR_KEY
```

### Run Analysis
```bash
# Note: For decoy analysis, use decoy_expensive or decoy_cheaper instead of decoy
python run_analysis.py --bias_name $BIAS_NAME --all_models $MODELS
```

### Few-shot Experiments
Add `--with_format_few_shot` or `--with_task_few_shot` flags when predicting and analyzing.

## Architecture

### Core Pipeline
1. **Data_generation/** - Creates experimental stimuli for each bias type
2. **Predict/** - Runs model inference on generated data
3. **Analysis/** - Computes bias scores and generates results

### Predictor System
All model predictors inherit from `Predict/Predictor.py` (abstract base class). To add a new model:
- Create a new predictor file inheriting from `Predictor` or `HFPredictor`
- Register the model in the appropriate list in `utils.py` (INSTRUCT_MODELS, VANILLA_MODELS, etc.)
- Add model loading logic in `Predict/predict.py:load_predictor()`

### Model Categories (utils.py)
- **INSTRUCT_MODELS**: Instruction-tuned models (Flan-T5, GPT-4, Llama-2-chat, Mistral-Instruct)
- **VANILLA_MODELS**: Base models without instruction tuning (T5, Llama-2, davinci)
- Model type determines prediction method (log-prob vs generation) and normalization settings

### Data Flow
- Generated data: `Data/{bias_name}/all_permutations/t_{templates}_{bias_types}_{Control|Treatment}.json`
- Predictions: `Predictions/{bias_name}/{product}/{model}/few_shot_{k}/{pred_type}/`
- Analysis outputs: CSV files with bias scores in prediction directories

### Key Configuration
- Templates define question formats per bias (configured in `Data_generation/templates.py`)
- Bias types: decoy (R, RF, F, R_EXTREAM), certainty (three_probs, two_probs), false_belief (dm_1, dm_2)
- Prediction modes: `predict_according_to_log_probs` (probability-based) vs generation (free-form text)
