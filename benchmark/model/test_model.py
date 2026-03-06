import csv
import os
import logging
from tqdm import tqdm
import argparse
from preprocess import setup_logging
from launcher import run_infer

BASELINE = ["hare", "megablocks", "Samoyeds", "transformers", "vllm"]
MODELS = ["deepseek", "MiniCPM", "mixtral-2x7B", "mixtral-2x11B", "qwen2_moe"]
REPOS = [
        "deepseek-ai/deepseek-moe-16b-base", 
        "openbmb/MiniCPM-MoE-8x2B", 
        "cloudyu/Mixtral_7Bx2_MoE_13B", 
        "cloudyu/Mixtral_11Bx2_MoE_19B",
        "Qwen/Qwen1.5-MoE-A2.7B"
    ]

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--seq_len', type=int, default=4096)

parser.add_argument('--mlp', action='store_true', default=False)
parser.add_argument('--layer', action='store_true', default=False)
parser.add_argument('--model', action='store_true', default=False)

parser.add_argument('--flash', action='store_true', default=False)

args = parser.parse_args()

def run_benchmark():
    # determine model type from flags
    if args.mlp:
        model_type = 'mlp'
    elif args.layer:
        model_type = 'layer'
    elif args.model:
        model_type = 'model'
    else:
        raise ValueError('Must specify one of --mlp, --layer, --model')

    flash_suffix = '_flash' if args.flash else ''

    output = f"result/b{args.batch_size}_seq{args.seq_len}_{model_type}{flash_suffix}.csv"
    logfile = f"log/b{args.batch_size}_seq{args.seq_len}_{model_type}{flash_suffix}.log"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    os.makedirs(os.path.dirname(logfile), exist_ok=True)

    setup_logging(logfile)
    logging.info(f"log save to {logfile}")
    logging.info(f"result save to {output}")
    
    csvfile = open(output, 'w', newline='')
    writer = csv.writer(csvfile)
    header = [
        "model_name", "model_type", "kernel_name", "iter", "batch_size", 
        "seq_len", "hidden_size", "intermediate_size", "num_experts", 
        "duration", "attn_implementation"
    ]
    writer.writerow(header)
    csvfile.flush()

    for j, (model, repo) in enumerate(zip(MODELS, REPOS)):
        for i, baseline in enumerate(BASELINE):
            result = run_infer(args.batch_size, args.seq_len, model, repo, baseline, model_type, args.flash)
            if result is None:
                message = [0] * len(header)
                message[0] = f"{model}"
                message[1] = f"{model_type}"
                message[2] = f"{baseline}"
            csv_row = list(result.values()) if result else message
            logging.info(",".join([str(i) for i in csv_row]))
            writer.writerow(csv_row)
            csvfile.flush()

    csvfile.close()

if __name__ == "__main__":
    run_benchmark()
