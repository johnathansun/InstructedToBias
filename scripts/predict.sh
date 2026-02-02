#!/bin/bash
#SBATCH --job-name=eval_bias_belief
#SBATCH --account=kempner_undergrads
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=0-06:00:00
#SBATCH --mem=256G
#SBATCH --output=scripts/logs/eval_bias_belief_output.out
#SBATCH --error=scripts/logs/eval_bias_belief_error.err
#SBATCH --mail-type=END
#SBATCH --mail-user=jlsun@college.harvard.edu

gpu=${1:-0}

module load python
mamba activate instructed_to_bias

CUDA_VISIBLE_DEVICES=$gpu python run_predict.py --bias_name false_belief --all_models Olmo-3-7B-Instruct

echo "Done with all traits"