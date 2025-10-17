#!/usr/bin/env python3
"""
Python script to dynamically generate and submit SLURM jobs for model training.
"""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path


def sanitize_model_name_for_path(model_name):
    """Convert model name to a filesystem-safe string."""
    return model_name.replace("/", "_").replace(":", "_").replace(" ", "_")


def generate_slurm_script(
    model_name,
    output_dir,
    job_name=None,
    partition="hopper-prod",
    gpus=1,
    time="1-00:00:00",
    target_column="edu_score",
    transformer_layers_unfrozen=0,
    samples_per_class=5000,
    lr=3e-4,
    **kwargs
):
    """Generate a SLURM script for the given model."""
    
    if job_name is None:
        safe_model_name = sanitize_model_name_for_path(model_name)
        job_name = f"classifier_{safe_model_name}"

    log_dir = os.path.join(output_dir, "logs")
    
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    # Build the Python command with all arguments
    python_cmd = f"""python classification/train_bert.py \\
    --base_model_name="{model_name}" \\
    --target_column="{target_column}" \\
    --output_dir="{output_dir}" \\
    --sample_per_class={samples_per_class} \\
    --transformer_layers_unfrozen={transformer_layers_unfrozen} \\
    --lr={lr}"""
    
    # Add any additional kwargs as arguments
    # Create logs folder
    for key, value in kwargs.items():
        if value is not None:
            python_cmd += f" \\\n    --{key}={value}"
    
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition {partition}
#SBATCH --gpus={gpus}
#SBATCH --output={log_dir}/%x_%j.log
#SBATCH --error={log_dir}/%x_%j.log
#SBATCH --time={time}

echo "Starting training for model: {model_name}"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
module load cuda/12.9

{python_cmd}

echo "Finished at: $(date)"
"""
    
    return slurm_script


def submit_job(model_name, base_output_dir, **kwargs):
    """Submit a SLURM job for the given model."""
    
    # Generate the SLURM script
    slurm_content = generate_slurm_script(model_name, base_output_dir, **kwargs)
    
    # Write to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.slurm', delete=False) as f:
        f.write(slurm_content)
        temp_script_path = f.name
    
    try:
        # Submit the job
        result = subprocess.run(
            ['sbatch', temp_script_path],
            capture_output=True,
            text=True,
            check=True
        )
        
        print(f"Successfully submitted job for model: {model_name}")
        print(f"Logs will be written to: {base_output_dir}/logs")
        print(f"SLURM output: {result.stdout.strip()}")
        
        return result.stdout.strip()
        
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job for model {model_name}:")
        print(f"Return code: {e.returncode}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise
    
    finally:
        # Clean up the temporary script
        os.unlink(temp_script_path)


def main():
    parser = argparse.ArgumentParser(description="Submit SLURM jobs for model training")
    
    # Required arguments
    parser.add_argument(
        "--base-model-name",
        type=str,
        help="Name of the model to train (e.g., 'Snowflake/snowflake-arctic-embed-m')"
    )
    
    # Optional SLURM configuration
    parser.add_argument("--partition", type=str, default="hopper-prod", help="SLURM partition")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--time", type=str, default="1-00:00:00", help="Time limit")
    parser.add_argument("--job-name", type=str, help="Custom job name (default: auto-generated)")
    
    # Training arguments
    parser.add_argument("--target-column", type=str, default="fw_edu_score", help="Target column for training")
    parser.add_argument("--dataset-name", type=str, default="HuggingFaceFW-Dev/fine-pdfs-classification-1-chunk-tb-teacher-1M-eng_Latn-Qwen_Qwen3-235B-A22B-Instruct-2507", help="Dataset name")
    parser.add_argument("--subset", type=str, help="Subset of the dataset")
    parser.add_argument("--checkpoint-dir", type=str, help="Checkpoint directory")
    parser.add_argument("--sample-per-class", type=str, default='64000', help="Samples per class")
    parser.add_argument("--transformer-layers-unfrozen", type=int, help="Number of transformer layers to unfreeze")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum number of steps")
    parser.add_argument("--prefix-run-name", type=str, help="Prefix for the run name")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    
    # Log directory
    parser.add_argument(
        "--log-dir", 
        type=str, 
        default="./training_logs",
        help="Base log directory"
    )
    
    args = parser.parse_args()
    
    # Prepare kwargs for training arguments
    training_kwargs = {}
    if args.dataset_name:
        training_kwargs["dataset_name"] = args.dataset_name
    if args.subset:
        training_kwargs["subset"] = args.subset
    if args.max_steps:
        training_kwargs["max_steps"] = args.max_steps
    if args.prefix_run_name:
        training_kwargs["prefix_run_name"] = args.prefix_run_name
    output_dir = f"{args.log_dir}/{args.base_model_name.replace('/', '_')}_{args.target_column.replace('-', '_')}_{args.sample_per_class}_{args.transformer_layers_unfrozen}_{args.lr}_{args.max_steps}"
    
    # Submit the job
    submit_job(
        model_name=args.base_model_name,
        base_output_dir=output_dir,
        job_name=args.job_name,
        partition=args.partition,
        transformer_layers_unfrozen=args.transformer_layers_unfrozen,
        gpus=args.gpus,
        time=args.time,
        target_column=args.target_column.replace("-", "_"),  # Convert back to underscore
        sample_per_class=args.sample_per_class,
        lr=args.lr,
        **training_kwargs
    )


if __name__ == "__main__":
    main()
