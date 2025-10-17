import argparse
from datetime import datetime
import os
import re
import subprocess
import tempfile
from typing import Optional, Set, List, Tuple
import random
import yaml
import json

from fsspec.core import url_to_fs
import itertools
from datatrove.io import get_datafolder
from loguru import logger


USER = os.environ.get("USER", "user")
EVAL_LOGS_PATH = f"/shared/{USER}/logs/pdf_project/experiments/evals/logs"
S3_EVALS_RESULTS_PREFIX = f"s3://shared-bucket/experiments/evals-test"
NANOTRON_PATH = "/shared/projects/new_training_setup/nanotron"
S5CMD_PATH = "/shared/projects/new_training_setup/training_venv/bin/s5cmd"

CPUS_PER_NODE = 88
GPUS_PER_NODE = 8
PARTITION = "hopper-prod"
NODES = 1

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

def parse_date(date_string: Optional[str]) -> Optional[datetime]:
    if date_string is None:
        return None
    try:
        return datetime.strptime(date_string, "%d-%m-%Y %H:%M:%S")
    except ValueError:
        raise ValueError("Invalid date format. Use 'DD-MM-YYYY HH:MM:SS'")


def checkpoint_exists(logging_dir: str, model_name: str, checkpoint: str, reference_date: Optional[datetime]) -> bool:
    s3_results_path = f"{S3_EVALS_RESULTS_PREFIX}/results/{model_name}/{checkpoint}/results_*"
    fs, path_prefix = url_to_fs(s3_results_path.rsplit('/', 1)[0])
    glob_pattern = s3_results_path.split('/')[-1]

    try:
        result_files = fs.glob(f"{path_prefix}/{glob_pattern}")
    except FileNotFoundError:
        result_files = []
    except Exception as e:
        logger.warning(f"Error accessing S3 path {s3_results_path}: {e}")
        result_files = []

    if len(result_files) == 0:
        return False

    if reference_date is None:
        return True

    timestamps = []
    for f in result_files:
        match = re.search(r'results_(.*)\.json$', f)
        if match:
            try:
                timestamps.append(datetime.strptime(match.group(1), "%Y-%m-%dT%H-%M-%S.%f"))
            except ValueError:
                logger.warning(f"Could not parse timestamp from filename: {f}")
        else:
             logger.warning(f"Could not extract timestamp from filename: {f}")

    return any(timestamp > reference_date for timestamp in timestamps)


def launch_slurm_job(launch_file_contents, debug, *args):
    """
        Small helper function to save a sbatch script and call it.
    Args:
        launch_file_contents: Contents of the sbatch script
        *args: any other arguments to pass to the sbatch command

    Returns: the id of the launched slurm job

    """
    with (tempfile.NamedTemporaryFile("w") if not debug else open("launch_file.sh", "w")) as f:
        f.write(launch_file_contents)
        f.flush()
        os.chmod(f.name, 0o755)
        if debug:
            # Report how to run the job locally
            print(f"To run the job locally, run:")
            print(f"bash {f.name}")
        else:
            try:
                return subprocess.check_output(["sbatch", *args, f.name]).decode("utf-8").split()[-1]
            except Exception as e:
                print(launch_file_contents, flush=True)
                raise e

def get_evaluated_tasks(model_name: str, checkpoint: str, reference_date: Optional[datetime]) -> Set[str]:
    """Get all tasks that have already been evaluated for a given model and checkpoint."""
    s3_results_path = f"{S3_EVALS_RESULTS_PREFIX}/results/{model_name}/{checkpoint}"
    fs, path_prefix = url_to_fs(s3_results_path)
    
    evaluated_tasks = set()
    try:
        # Get all JSON files in the checkpoint results directory
        result_files = fs.glob(f"{path_prefix}/results_*.json")
    except FileNotFoundError:
        return set()

    for result_file in result_files:
        try:
            # Check if file is newer than reference date if provided
            if reference_date is not None:
                match = re.search(r'results_(.*)\.json$', result_file)
                if match:
                    try:
                        file_timestamp = datetime.strptime(match.group(1), "%Y-%m-%dT%H-%M-%S.%f")
                        if file_timestamp <= reference_date:
                            continue  # Skip files older than reference date
                    except ValueError:
                        logger.warning(f"Could not parse timestamp from filename: {result_file}")
                        continue
            
            # Read the JSON file and extract task names from results
            with fs.open(result_file, 'r') as f:
                data = json.load(f)
                if 'results' in data:
                    tasks = [x.replace(":_average", "") for x in data['results'].keys()]
                    evaluated_tasks.update(tasks)
                    
        except Exception as e:
            logger.warning(f"Error reading result file {result_file}: {e}")
    
    return evaluated_tasks

def get_tasks_from_file(tasks_list_path: str) -> dict[tuple[str, str, str], str]:
    """
    Reads all task names from a tasks list file.
    
    Args:
        tasks_list_path: Path to the tasks list file.
    
    Returns:
        Set of task names.
    """
    tasks = dict()
    try:
        with open(tasks_list_path, 'r') as f:
            for line in f:
                line_stripped = line.strip()
                if line_stripped and not line_stripped.startswith('#'):
                    # Remove the last part separated by |
                    parts = '|'.join(line_stripped.split('|')[:-1])
                    tasks[parts] = line_stripped
    except FileNotFoundError:
        logger.error(f"Tasks list file not found: {tasks_list_path}")
    except Exception as e:
        logger.error(f"Error reading tasks from {tasks_list_path}: {e}")
    return tasks

def get_checkpoints_to_run(s3_path: str, model_name: str, checkpoints: str, logging_dir: str, overwrite: bool = False,
                           after_date: Optional[str] = None, tasks_list_path: Optional[str] = None) -> List[Tuple[str, Set[str]]]:
    """
    Retrieves checkpoints to run and their remaining tasks.
    
    Args:
        s3_path: S3 path to the model checkpoints.
        model_name: Name of the model (e.g., "1p46G-control-english-fw-ft-bl-28BT-seed-6").
        checkpoints: Comma-separated list of checkpoints to run, or "all".
        logging_dir: S3 path to push results to.
        overwrite: If True, overwrite existing results.
        after_date: Only consider checkpoints newer than this date (DD-MM-YYYY HH:MM:SS).
        tasks_list_path: Path to the original tasks file.
    
    Returns:
        List of tuples (checkpoint_name, remaining_tasks) for checkpoints to evaluate.
    """
    reference_date = parse_date(after_date)
    df = get_datafolder(s3_path)
    try:
        avail_checkpoints = [i for i in sorted(df.ls("", detail=False)) if i != "latest.txt"]
    except FileNotFoundError:
        logger.error(f"No checkpoints found in {s3_path}")
        avail_checkpoints = []
    logger.info(f"Found {len(avail_checkpoints)} checkpoints in {s3_path}: {avail_checkpoints}")
    selected_checkpoints = checkpoints.split(",") if checkpoints != "all" else avail_checkpoints
    not_found_checkpoints = [ckpt for ckpt in selected_checkpoints if ckpt not in avail_checkpoints]
    if len(not_found_checkpoints) > 0:
        raise ValueError(f"Checkpoints not found in \"{s3_path}\": {not_found_checkpoints}")

    # if not overwrite:
    #     completed_checkpoints = [
    #         ckpt for ckpt in selected_checkpoints
    #         if checkpoint_exists(logging_dir, model_name, ckpt, reference_date)
    #     ]
    #     completed = len(completed_checkpoints)
    #     selected_checkpoints = list(set(selected_checkpoints) - set(completed_checkpoints))
    #     if completed:
    #         logger.info(f"Skipping {completed} already evaluated checkpoints.")

    checkpoints_with_tasks = []
    tasks_from_file = get_tasks_from_file(tasks_list_path) if tasks_list_path else None
    if tasks_list_path:
        for ckpt in selected_checkpoints:
            get_checkpoint_evaluated_tasks = get_evaluated_tasks(model_name, ckpt, reference_date)
            remaining_tasks = set(tasks_from_file.keys()) - get_checkpoint_evaluated_tasks
            tasks_str = ",".join([tasks_from_file[task] for task in remaining_tasks])
            if tasks_str:
                checkpoints_with_tasks.append((ckpt, tasks_str))
    return checkpoints_with_tasks


def read_tasks_from_file(tasks_list_path: str) -> Set[str]:
    """
    Reads all task names from a tasks list file.
    
    Args:
        tasks_list_path: Path to the tasks list file.
    
    Returns:
        Set of task names.
    """
    tasks = set()
    try:
        with open(tasks_list_path, 'r') as f:
            for line in f:
                line_stripped = line.strip()
                if line_stripped and not line_stripped.startswith('#'):
                    parts = line_stripped.split('|')
                    if len(parts) >= 2:
                        task_name = parts[1]
                        tasks.add(task_name)
    except FileNotFoundError:
        logger.error(f"Tasks list file not found: {tasks_list_path}")
    except Exception as e:
        logger.error(f"Error reading tasks from {tasks_list_path}: {e}")
    return tasks




parser = argparse.ArgumentParser("Launch evals for a set of checkpoints.")

parser.add_argument(
    "model_name", type=str,
    help="Model name on s3. Example: 1p46G-control-english-fw-ft-bl-28BT-seed-6. Use commas for multiple models"
)
parser.add_argument(
    "--s3_prefix", type=str, help="s3://path/to/models/ by default",
    default="s3://shared-bucket/experiments/checkpoints"
)
parser.add_argument(
    "--checkpoints", "-ckpts", type=str, help="Comma separated list of checkpoints to run, or \"all\"",
    default="all"
)
parser.add_argument(
    "--model-template", type=str, help="Template to use for the model name",
    default="{model_name}"
)
parser.add_argument("--run-all", action="store_true", default=False, help="Run in sequence")

parser.add_argument("--tasks", type=str, help="Comma separated list of tasks to run, or \"all\"",
                    default="/admin/home/user/shared/projects/new_training_setup/tasks.txt")
parser.add_argument("--custom-tasks", type=str, help="lighteval custom tasks",
                    default="/admin/home/user/shared/projects/new_training_setup/task_list.py")
parser.add_argument(
    "--offline-datasets", action="store_true", help="Turns off datasets downloading", default=False
)
parser.add_argument(
    "--seed", help="Defines seeds to use in model template. Comma separated list of seeds", default="6"
)
parser.add_argument("--qos", type=str, default="normal", help="qos to use")
parser.add_argument("--time_limit", type=str, default="1:50:00", help="slurm time limit. 1:50:00 by default")
parser.add_argument("--parallel", "-p", type=int, default=5, help="How many eval tasks to run simultaneously")
parser.add_argument("--batch_size", "-bs", type=int, default=None, help="Batch size")
parser.add_argument("--gpus", "-g", type=int, default=GPUS_PER_NODE, help="How many gpus to use")
parser.add_argument("--logging_dir", type=str, default=S3_EVALS_RESULTS_PREFIX,
                    help="S3 repo to push results to")
parser.add_argument("-d", help="dependency job", type=str, default=None)
parser.add_argument("--overwrite", "-ow", action="store_true", default=False,
                    help="Overwrite existing eval results. Will skip completed checkpoints by default")
parser.add_argument("--after-date", type=str, default=None,
                    help="Only consider checkpoints newer than this date (DD-MM-YYYY HH:MM:SS)")
parser.add_argument("--job-prefix", type=str, default="", help="Prefix to add to the job name")
parser.add_argument("--debug", action="store_true", default=False)

if __name__ == "__main__":
    args = parser.parse_args()

    job_id = None
    for model_name, seed in itertools.product(args.model_name.split(","), args.seed.split(",")):
        formatted_model_name = args.model_template.format(model_name=model_name, seed=seed)
        # get language from the model_name
        language = "_".join(formatted_model_name.split("_")[:2])
        custom_tasks_path = custom_tasks.get(language)
        tasks_list_path = tasks_list.get(language)

        if custom_tasks_path is None and tasks_list_path is None:
            print(f"Language {language} not found in custom_tasks or tasks_list, defaulting to English")
            custom_tasks_path = custom_tasks.get("eng_Latn")
            tasks_list_path = tasks_list.get("eng_Latn")

        s3_path = args.s3_prefix.removesuffix("/") + "/" + formatted_model_name if not formatted_model_name.startswith(
            "s3://") else formatted_model_name
        
        # Get checkpoints with their remaining tasks
        checkpoints_with_tasks = get_checkpoints_to_run(
            s3_path, formatted_model_name, args.checkpoints, args.logging_dir,
            overwrite=args.overwrite, after_date=args.after_date, tasks_list_path=tasks_list_path
        )
        
        logger.info(f"Found {len(checkpoints_with_tasks)} checkpoints to evaluate for {formatted_model_name}")
        if not checkpoints_with_tasks:
            print(f"No checkpoints to run for {formatted_model_name}.")
            continue

        # Create the same folder structure as in one_job_runner.py but with multi-step
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{timestamp}-eval_{formatted_model_name}".replace(" ", "_")
        
        # Create paths following one_job_runner.py structure
        eval_launch_script_path = os.path.join(EVAL_LOGS_PATH, formatted_model_name, "launch_scripts", "multi-step")
        eval_logs_path = os.path.join(EVAL_LOGS_PATH, formatted_model_name, "logs", "multi-step")
        model_config_path = os.path.join(EVAL_LOGS_PATH, formatted_model_name, "model_config", "multi-step")
        
        # Create directories
        os.makedirs(eval_launch_script_path, exist_ok=True)
        os.makedirs(eval_logs_path, exist_ok=True)
        os.makedirs(model_config_path, exist_ok=True)

        # Prepare bash arrays for checkpoints and their corresponding tasks
        bash_ckpts_list = "(" + " ".join(f'"{item[0]}"' for item in checkpoints_with_tasks) + ")"
        bash_tasks_list = "(" + " ".join(f'"{item[1]}"' for item in checkpoints_with_tasks) + ")"

        # Create lighteval config yaml
        lighteval_config_yaml = {
            "model_parameters": {
                "model_name": "$LOCAL_DOWNLOAD_CHECKPOINT_FOLDER/hf_model",
                "batch_size": args.batch_size if args.batch_size is not None else None,
                "trust_remote_code": True,
                "dtype": "bfloat16",
                "generation_parameters": {
                    "repetition_penalty": 1.2,
                }
            }
        }

        # Write the config to file
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        lighteval_launch_config_path = os.path.join(model_config_path, f"model_config-{current_time}.yaml")
        with open(lighteval_launch_config_path, "w") as f:
            yaml.dump(lighteval_config_yaml, f)

        deps = []
        if args.d:
            deps.append(f"afterok:{args.d}")
        if args.run_all and job_id:
            deps.append(f"afterany:{job_id}")

        master_port = 1024 + random.randint(0, 64511)
        
        # Count total tasks to be evaluated for job name
        total_remaining_tasks = sum(len(tasks.split(',')) for _, tasks in checkpoints_with_tasks)
        all_possible_tasks = len(read_tasks_from_file(tasks_list_path))
        task_suffix = f"_{total_remaining_tasks}tasks" if total_remaining_tasks < all_possible_tasks * len(checkpoints_with_tasks) else ""

        launch_script = f"""#!/bin/bash
#SBATCH --job-name={args.job_prefix}eval-{formatted_model_name}{task_suffix}
#SBATCH --nodes={NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --partition={PARTITION}
{f'#SBATCH --qos={args.qos}' if args.qos else ''}
#SBATCH --array=0-{len(checkpoints_with_tasks) - 1}%{args.parallel}
#SBATCH --gres=gpu:{args.gpus}
#SBATCH --time={args.time_limit}
#SBATCH --cpus-per-task={CPUS_PER_NODE}
#SBATCH --output={eval_logs_path}/eval-%A_%a.out
#SBATCH --error={eval_logs_path}/eval-%A_%a.out
{"#SBATCH --dependency=" + ",".join(deps) if deps else ""}
#SBATCH --requeue
###########################################

# Ensure cache is on shared not on admin
# export HF_DATASETS_OFFLINE={1 if args.offline_datasets else 0}
export TMPDIR=/scratch/{USER}/tmp
source /admin/home/user/shared/projects/new_training_setup/training_venv/bin/activate
mkdir -p $TMPDIR

###########################################

set -x -e
echo "START TIME: $(date)"
echo python3 version = `python3 --version`

# SLURM stuff
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT={master_port}
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

# CUDA/NCCL Settings
export CUDA_DEVICE_MAX_CONNECTIONS="1"

module purge
module load cuda/12.4

echo "Running on $COUNT_NODE nodes: $HOSTNAMES"

# Bash arrays for checkpoints and their corresponding tasks
CHECKPOINTS_LIST={bash_ckpts_list}
TASKS_LIST={bash_tasks_list}

# Get checkpoint and tasks for this array task
NSTEP=$((SLURM_ARRAY_TASK_ID))
STEP=${{CHECKPOINTS_LIST[$NSTEP]}}
TASKS_TO_EVAL=${{TASKS_LIST[$NSTEP]}}

echo "Processing checkpoint: $STEP"
echo "Tasks to evaluate: $TASKS_TO_EVAL"

RANDOM_NUMBER=$((RANDOM % 60 + 5))
# Local directory for downloading the checkpoint
LOCAL_DOWNLOAD_CHECKPOINT_FOLDER=/scratch/{USER}/eval-checkpoints/{formatted_model_name}-$RANDOM_NUMBER/$STEP
mkdir -p $LOCAL_DOWNLOAD_CHECKPOINT_FOLDER

# Copying checkpoint from s3 to the node's scratch space
echo "Downloading checkpoint step $STEP from {s3_path} to $LOCAL_DOWNLOAD_CHECKPOINT_FOLDER"
{S5CMD_PATH} --stat cp \\
    --concurrency 50 \\
    --exclude "optimizer/*" \\
    --exclude "random/*" \\
    --exclude "lr_scheduler/*" \\
    --part-size 100 \\
    "{s3_path}/$STEP/*" "$LOCAL_DOWNLOAD_CHECKPOINT_FOLDER/"

# Convert nanotron checkpoint to huggingface format
# First remember the current working directory, then change to the nanotron path, call the conversion script, then change back to the original directory
# Save current directory
CURRENT_DIR=$(pwd)

# Change to nanotron directory
cd {NANOTRON_PATH}

# Run conversion script
torchrun --standalone --nproc_per_node=1 -m examples.llama.convert_nanotron_to_hf \\
    --checkpoint_path $LOCAL_DOWNLOAD_CHECKPOINT_FOLDER \\
    --save_path $LOCAL_DOWNLOAD_CHECKPOINT_FOLDER/hf_model \\
    --check_conversion

# Change back to original directory
cd $CURRENT_DIR

# Create lighteval config yaml for this run
LIGHTEVAL_CONFIG_PATH="$LOCAL_DOWNLOAD_CHECKPOINT_FOLDER/lighteval_config.yaml"
cat > $LIGHTEVAL_CONFIG_PATH << EOL
model_parameters:
  model_name: $LOCAL_DOWNLOAD_CHECKPOINT_FOLDER/hf_model
  batch_size: {args.batch_size if args.batch_size is not None else 'null'}
  trust_remote_code: true
  dtype: bfloat16
  model_name_override: {formatted_model_name}/$STEP
  generation_parameters:
    repetition_penalty: 1.2
EOL

# Launch lighteval using accelerate
echo "Running evaluation for checkpoint $STEP with tasks: $TASKS_TO_EVAL"
CUDA_DEVICE_MAX_CONNECTIONS=1 accelerate launch \\
    --multi_gpu \\
    --num_processes {args.gpus} \\
    -m lighteval accelerate \\
    --custom-tasks {custom_tasks_path} \\
    --dataset-loading-processes {CPUS_PER_NODE} \\
    --max-samples 1000 \\
    --output-dir {args.logging_dir} \\
    --save-details \\
    $LIGHTEVAL_CONFIG_PATH \\
    "$TASKS_TO_EVAL"

# Clean up downloaded checkpoint from scratch
echo "Cleaning up downloaded checkpoint $STEP..."
rm -rf "$LOCAL_DOWNLOAD_CHECKPOINT_FOLDER"
echo "Cleanup complete."

echo "END TIME: $(date)"
"""
        launched_id = launch_slurm_job(launch_script, debug=args.debug)
        
        # Create summary of what's being evaluated
        task_summary = []
        for ckpt, tasks in checkpoints_with_tasks:
            task_summary.append(f"  {ckpt}: {len(tasks.split(','))} tasks")
        
        logger.success(
            f"{formatted_model_name} evals launched with id={launched_id}. "
            f"Total: {len(checkpoints_with_tasks)} checkpoints, {total_remaining_tasks} tasks remaining.\n"
            f"Details:\n" + "\n".join(task_summary) + f"\nLocal logs: {eval_logs_path}")
        job_id = launched_id