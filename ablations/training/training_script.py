#!/usr/bin/env python3
import os
import argparse
import random
import tempfile
import subprocess
import yaml
from pathlib import Path
from datetime import datetime

# Constants - modify these as needed
USER = os.environ.get('USER')
LOCAL_TMP_PATH_ON_NODE = f"/scratch/{USER}"
TRAINING_LOGS_PATH = f"/shared/{USER}/logs/pdf_project/experiments/training/logs"
EVAL_LOGS_PATH = f"/shared/{USER}/logs/pdf_project/experiments/evals"
NANOTRON_PATH = f"/shared/user/projects/new_training_setup/nanotron"
S5CMD_PATH = "/shared/user/projects/new_training_setup/training_venv/bin/s5cmd"
S3_CHECKPOINTS_PREFIX = "s3://bucket/experiments/checkpoints"
LAUNCH_CONFIGS_PATH = f"/shared/{USER}/logs/pdf_project/experiments/training/launch-configs"
SLURM_SCRIPT_PATH = f"/shared/{USER}/logs/pdf_project/experiments/training/slurm-scripts"
EVALS_OUTPUT_PATH = "s3://bucket/experiments/evals-test"
NUM_GPUS = 8
NUM_CPUS_IN_NODE = 88

# Default model config (updated with configuration from old_training_debug.py)
MODEL_CONFIG = f"""
checkpoints:
  checkpoint_interval: 500
  checkpoints_path: {LOCAL_TMP_PATH_ON_NODE}/checkpoints
  checkpoints_path_is_shared_file_system: false
  load_lr_scheduler: true
  load_optimizer: true
  resume_checkpoint_path: null
  save_final_state: true
  save_initial_state: false
data_stages:
- data:
    dataset:
      pad_samples_to_global_batch_size: false
      return_positions: true
      token_size_in_bytes: 4
      use_old_brrr_dataloader: false
      tokenizer_name: hynky/Llama-3.2-1B-no-bos
      vocab_size: 128256
    num_loading_workers: 0
    seed: 6
  name: stable
  start_training_step: 1
general:
  benchmark_csv_path: null
  consumed_train_samples: null
  ignore_sanity_checks: true
  project: fine-pdfs
  run: tmp
  seed: 6
  step: null
logging:
  iteration_step_info_interval: 5
  log_level: info
  log_level_replica: info
model:
  ddp_bucket_cap_mb: 50
  dtype: bfloat16
  init_method:
    std: 0.02
  make_vocab_size_divisible_by: 1
  model_config:
    _attn_implementation: flash_attention_2
    _fused_rms_norm: true
    _fused_rotary_emb: true
    _use_doc_masking: true
    _use_qkv_packed: true
    attention_bias: false
    bos_token_id: 128000
    eos_token_id: 128001
    flex_attention_mask: null
    hidden_act: silu
    hidden_size: 2048
    initializer_range: 0.02
    intermediate_size: 6144
    is_qwen2_config: true
    max_position_embeddings: 4096
    moe_config: null
    num_attention_heads: 16
    num_hidden_layers: 28
    num_key_value_heads: 8
    pad_token_id: null
    pretraining_tp: 1
    rms_norm_eps: 1.0e-06
    rope_interleaved: false
    rope_scaling: null
    rope_theta: 10000
    sliding_window_size: null
    tie_word_embeddings: true
    use_cache: true
    vocab_size: 128256
    z_loss_coefficient: 1.0e-05
    z_loss_enabled: false
    no_rope_layer: null
optimizer:
  accumulate_grad_in_fp32: true
  clip_grad: 1.0
  learning_rate_scheduler:
    learning_rate: 5e-4
    lr_decay_starting_step: 2861
    lr_decay_steps: 1
    lr_decay_style: cosine
    lr_warmup_steps: 2861
    lr_warmup_style: linear
    min_decay_lr: 0
  optimizer_factory:
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_eps: 1.0e-08
    name: adamW
    torch_adam_is_fused: true
  weight_decay: 0.1
  weight_decay_exclude_named_params:
  - .*token_embedding.*
  zero_stage: 0
parallelism:
  context_parallel_size: 1
  dp: 64
  expert_parallel_size: 1
  pp: 1
  pp_engine: 1f1b
  recompute_layer: false
  tp: 1
  tp_linear_async_communication: true
  tp_mode: REDUCE_SCATTER
  tp_recompute_allgather: true
profiler: null
s3_upload:
  remove_after_upload: true
  s5cmd_concurrency: 5
  s5cmd_numworkers: 16
  s5cmd_path: {S5CMD_PATH}
  upload_s3_path: {S3_CHECKPOINTS_PREFIX}
tokenizer:
  tokenizer_max_length: 4096
  tokenizer_name_or_path: user/Llama-3.2-1B-no-bos
  tokenizer_revision: null
metrics_logging:
  log_level: 1
  log_detail_interval: 200
tokens:
  batch_accumulation_per_replica: 4
  limit_test_batches: 0
  limit_val_batches: 0
  micro_batch_size: 2
  sequence_length: 4096
  train_steps: null
  val_check_interval: 0
lighteval:
  output_dir: {EVALS_OUTPUT_PATH}
  logs_path: {EVAL_LOGS_PATH}
  local_checkpoint_dir: {LOCAL_TMP_PATH_ON_NODE}/evals-ckpt
  upload_to_wandb: false
  eval_interval: 500
  eval_interval_file: null
  nanotron_path: {NANOTRON_PATH}
  batch_size: 8
  slurm:
    gpus_per_node: 8
    hf_cache: "/shared/user/.cache/huggingface"
    partition: "hopper-prod"
    cpus_per_task: 88
    qos: "normal"
    time: "01:59:00"
  tasks:
    tasks: /admin/home/user/shared/projects/new_training_setup/tasks.txt
    custom_tasks: /admin/home/user/shared/projects/new_training_setup/task_list.py
    max_samples: 1000
"""

custom_tasks = {
  "eng_Latn": "/admin/home/user/shared/projects/new_training_setup/task_list.py",
  "fra_Latn": "lighteval.tasks.multilingual.tasks",
  "arb_Arab": "lighteval.tasks.multilingual.tasks",
  "cmn_Hani": "lighteval.tasks.multilingual.tasks",
  "rus_Cyrl": "lighteval.tasks.multilingual.tasks",
}

tasks_list = {
  "eng_Latn": "/admin/home/user/shared/projects/new_training_setup/tasks_txt/tasks_eng.txt",
  "fra_Latn": "/admin/home/user/shared/projects/new_training_setup/tasks_txt/tasks_fra.txt",
  "arb_Arab": "/admin/home/user/shared/projects/new_training_setup/tasks_txt/tasks_ara.txt",
  "cmn_Hani": "/admin/home/user/shared/projects/new_training_setup/tasks_txt/tasks_zho.txt",
  "rus_Cyrl": "/admin/home/user/shared/projects/new_training_setup/tasks_txt/tasks_rus.txt",
}

def launch_slurm_job(launch_file_contents, job_id, nodes, background, run_name, timestamp, *args):
    """
    Small helper function to save a sbatch script and call it.
    Args:
        launch_file_contents: Contents of the sbatch script
        *args: any other arguments to pass to the sbatch command

    Returns: the id of the launched slurm job
    """
    os.makedirs(f"{SLURM_SCRIPT_PATH}/{run_name}", exist_ok=True)
    with open(f"{SLURM_SCRIPT_PATH}/{run_name}/{job_id}-{timestamp}.sh", "w") as f:
        f.write(launch_file_contents)
        f.flush()
      
    # Make the script executable
    os.chmod(f.name, 0o755)
    if job_id:
      srun_args = ["srun", "--jobid", job_id, "--ntasks-per-node", "1", "--nodes", str(nodes)]
      if background:
        srun_args += ["--output", f"{TRAINING_LOGS_PATH}/{run_name}/train-{timestamp}.out", "--error", f"{TRAINING_LOGS_PATH}/{run_name}/train-{timestamp}.err"]
        # Run the job in background - DOES NOT WAIT
        subprocess.Popen(srun_args + list(args) + [f.name])
        print(f"Running in background. Logs: {TRAINING_LOGS_PATH}/{run_name}/train-{timestamp}.out {TRAINING_LOGS_PATH}/{run_name}/train-{timestamp}.err")
      else:
        # Run in foreground till job is done - WAITS
        subprocess.check_call(srun_args + list(args) + [f.name])
    else:
      return subprocess.run(["sbatch", *args, f.name], capture_output=True, text=True).stdout.split()[-1]

def main():
    parser = argparse.ArgumentParser(description="Launch training job with updated configuration")
    parser.add_argument("data", help="Dataset folder path (can be S3 path)", type=str)
    parser.add_argument("run_name", help="Run name", type=str)
    parser.add_argument("--tokenizer", help="Tokenizer name or path", type=str, default="user/Llama-3.2-1B-no-bos")
    parser.add_argument("-d", help="Dependency job", type=str, default=None)
    parser.add_argument("--seed", help="Seed", type=int, default=6)
    parser.add_argument("--data-seed", help="Data seed", type=int, default=6)
    parser.add_argument("--train_steps", "-ts", help="Training steps", type=int, default=17_000)
    parser.add_argument("--priority", "--qos", "-p", help="QoS to use", type=str, default="normal")
    parser.add_argument("--nodes", help="Number of nodes", type=int, default=1)
    parser.add_argument("--debug", help="Enable d/ntuebug mode", action="store_true")
    parser.add_argument("--job_id", help="Job ID", type=str, default=None)
    parser.add_argument("--lr", help="Learning rate", type=float, default=5e-4)
    parser.add_argument("--background", help="Run in background", action="store_true")
    parser.add_argument("--reservation", help="SLURM reservation name", type=str, default=None)
    parser.add_argument("--time", help="SLURM time", type=str, default="08:59:00")
    parser.add_argument("--resume", help="Set resume checkpoint path to the checkpoint path", action="store_true")
    args = parser.parse_args()
    
    # Load the config
    config = yaml.safe_load(MODEL_CONFIG)

    # Tokens per step == 4096 * batch_accumulation_per_replica * micro_batch_size * dp
    total_tokens_consumed = round(4096 * 4 * 2 * NUM_GPUS * args.nodes * args.train_steps / 1e9) # in billions
    print(f"Total tokens consumed: {total_tokens_consumed}B")
    
    # Update the config with the provided arguments
    run_name = args.run_name.replace(" ", "_")
    run_name = f"{run_name}-{total_tokens_consumed}B-seed-{args.seed + (args.data_seed * 100)}"
    
    # Calculate local dataset path if using S3
    local_dataset_path = f"{LOCAL_TMP_PATH_ON_NODE}/dataset/{run_name}/"
    
    # Update data_stages with dynamically added dataset configuration
    dataset_folder = local_dataset_path if args.data.startswith("s3://") else args.data
    
    # Dynamically add dataset entries
    config["data_stages"][0]["data"]["dataset"]["dataset_folder"] = [dataset_folder]
    config["data_stages"][0]["data"]["dataset"]["dataset_weights"] = [1.0]
    
    # Update general config
    config["general"]["run"] = run_name
    config["general"]["seed"] = args.seed
    config["data_stages"][0]["data"]["seed"] = args.data_seed
    
    # Update tokenizer
    config["tokenizer"]["tokenizer_name_or_path"] = args.tokenizer
    
    # Update checkpoint paths
    checkpoint_path = f"{LOCAL_TMP_PATH_ON_NODE}/checkpoints/{run_name}"
    config["checkpoints"]["checkpoints_path"] = checkpoint_path
    config["s3_upload"]["upload_s3_path"] = f"{S3_CHECKPOINTS_PREFIX}/{run_name}"
    
    # Set resume checkpoint path if --resume flag is provided
    if args.resume:
        config["checkpoints"]["resume_checkpoint_path"] = f"{S3_CHECKPOINTS_PREFIX}/{run_name}"
    
    # Update training steps
    config["tokens"]["train_steps"] = args.train_steps

    # Lighteval config
    # get language from the run_name
    language = run_name.split("-")[0]
    custom_tasks_path = custom_tasks.get(language)
    tasks_list_path = tasks_list.get(language)

    # Default to English if language not found
    if custom_tasks_path is None and tasks_list_path is None:
        print(f"Language {language} not found in custom_tasks or tasks_list, defaulting to English")
        custom_tasks_path = custom_tasks.get("eng_Latn")
        tasks_list_path = tasks_list.get("eng_Latn")

    config["lighteval"]["tasks"]["custom_tasks"] = custom_tasks_path
    config["lighteval"]["tasks"]["tasks"] = tasks_list_path

    
    # Debug mode settings
    if args.debug:
        config["parallelism"]["dp"] = 2
        config["parallelism"]["pp"] = 2
        config["parallelism"]["tp"] = 2
        config["tokens"]["micro_batch_size"] = 3
        config["tokens"]["batch_accumulation_per_replica"] = 2
        args.nodes = 1
      
    else:
      config["parallelism"]["dp"] = args.nodes * NUM_GPUS

    # Update training steps
    config["tokens"]["train_steps"] = args.train_steps

    # Update decay start to 10% of training steps
    decay_start_step = round(args.train_steps * 0.1)
    decay_steps = args.train_steps - decay_start_step
    config["optimizer"]["learning_rate_scheduler"]["lr_decay_steps"] = decay_steps
    config["optimizer"]["learning_rate_scheduler"]["lr_warmup_steps"] = decay_start_step
    config["optimizer"]["learning_rate_scheduler"]["lr_decay_starting_step"] = decay_start_step

    # Update learning rate
    config["optimizer"]["learning_rate_scheduler"]["learning_rate"] = args.lr

    # Update decay learning rate
    min_decay_lr = args.lr / 10
    config["optimizer"]["learning_rate_scheduler"]["min_decay_lr"] = min_decay_lr
    
    # Tokens per step == 4096 * batch_accumulation_per_replica * micro_batch_size * dp
    # 4096 * 4 * 2 * 64 = 2097152
    
    # Save the updated config
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(f"{LAUNCH_CONFIGS_PATH}/{run_name}", exist_ok=True)
    config_path_yaml = f"{LAUNCH_CONFIGS_PATH}/{run_name}/{timestamp}.yaml"
    
    with open(config_path_yaml, "w") as f:
        yaml.dump(config, f)
    
    # Create logs directory
    os.makedirs(f"{TRAINING_LOGS_PATH}/{run_name}", exist_ok=True)
    
    # Build dataset download command if needed
    dataset_download_cmd = ""
    run_cmd = "" if args.job_id else "srun"
    if args.data.startswith("s3://"):
        dataset_download_cmd = f"""
{run_cmd} rm -rf {LOCAL_TMP_PATH_ON_NODE}/dataset
{run_cmd} {S5CMD_PATH} cp '{args.data.removesuffix("/")}/*' {local_dataset_path}
# """
    
    # Build SLURM job script
    job_name = run_name
    
    sbatch_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={args.nodes}
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task={NUM_CPUS_IN_NODE}
#SBATCH --gres=gpu:{NUM_GPUS}
#SBATCH --partition=hopper-prod
#SBATCH --output={TRAINING_LOGS_PATH}/{run_name}/train-{timestamp}-%x-%j
#SBATCH --qos={args.priority}
#SBATCH --begin=now+0minutes
#SBATCH --time={args.time}
#SBATCH --exclusive
{"#SBATCH --dependency=afterok:" + args.d if args.d else ""}
{"#SBATCH --reservation=" + args.reservation if args.reservation else ""}

set -x -e

echo "START TIME: $(date)"
secs_to_human(){{
    echo "$(( ${{1}} / 3600 )):$(( (${{1}} / 60) % 60 )):$(( ${{1}} % 60 ))"
}}
start=$(date +%s)
echo "$(date -d @${{start}} "+%Y-%m-%d %H:%M:%S"): ${{SLURM_JOB_NAME}} start id=${{SLURM_JOB_ID}}"

{dataset_download_cmd}

# SLURM setup
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT={1024 + random.randint(0, 64511)}
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

export TMPDIR={LOCAL_TMP_PATH_ON_NODE}
export CUDA_DEVICE_MAX_CONNECTIONS="1"

module load cuda/12.4

echo go $COUNT_NODE
echo $HOSTNAMES

CMD=" \
    {NANOTRON_PATH}/run_train.py \
    --config-file {config_path_yaml}
    "
export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node {NUM_GPUS} \
    --nnodes $COUNT_NODE \
    --rdzv-backend c10d \
    --rdzv-endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv-id $SLURM_JOB_ID \
    --node_rank $SLURM_PROCID \
    --role $SLURMD_NODENAME: \
    --max_restarts 0 \
    --tee 3 \
    "

# Add small random delay to avoid concurrent hub requests
random_milliseconds=$(( RANDOM % 1001 ))
sleep_time=$(bc <<< "scale=3; $random_milliseconds / 1000")
echo "Sleeping for $sleep_time seconds..."
sleep $sleep_time

{run_cmd} bash -c "$LAUNCHER $CMD"
echo "END TIME: $(date)"

# Clean up dataset if downloaded from S3
{
    "" if not args.data.startswith("s3://") else f"{run_cmd} rm -rf {local_dataset_path}"
}
"""
    
    # Launch the job
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    job_id = launch_slurm_job(sbatch_script, args.job_id, args.nodes, args.background, run_name, timestamp)
    log_path = f"{TRAINING_LOGS_PATH}/{run_name}/train-{timestamp}-{job_name}-{job_id}"
    
    print(f"Launched with Slurm job id={job_id}")
    print(f"To view the logs, use the command: tail -f {log_path}")

if __name__ == "__main__":
    main()