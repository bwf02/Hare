import os
import sys
import subprocess
import logging
CURRENT_FILE_PATH = os.path.abspath(__file__)
WORK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE_PATH)))
# print(WORK_DIR)
sys.path.extend([f"{WORK_DIR}/hare/infer", f"{WORK_DIR}/third_party/megablocks", f"{WORK_DIR}/hare"])

from preprocess import parse_benchmark_data

def run_command(cmd):
    logger = logging.getLogger("file")
    logger.info(f"command: {cmd}")
    result = subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    if result.returncode != 0:
        logger.error(f"Error: {result.stdout}\nfailed with code {result.returncode}")
        logger.error(f"Error: {cmd}\nfailed with code {result.returncode}")
        logger.error(f"stderr: {result.stderr}")
        return None
    return result

def run_infer(batch_size, seq_len, model, repo, baseline, model_type, use_flash):
    if baseline == "hare":
        cmd = f"cd {WORK_DIR}/hare/infer && "
    else:
        cmd = f"cd {WORK_DIR}/third_party/Samoyeds && "

    if use_flash:
        cmd += f"python {model.split('-')[0]}_{baseline}.py --repo {repo} --batch_size {batch_size} --seq_len {seq_len} --{model_type} --flash --time --huggingface_config"
    else:
        cmd += f"python {model.split('-')[0]}_{baseline}.py --repo {repo} --batch_size {batch_size} --seq_len {seq_len} --{model_type} --time --huggingface_config"
    result = run_command(cmd)
    if result is None:
        return None
    return parse_benchmark_data(result.stdout)