# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import time
import json
import fire

from pathlib import Path
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from human_eval.evaluation import evaluate_functional_correctness
from llama import ModelArgs, Transformer, Tokenizer, LLaMA
from human_eval.data import write_jsonl, read_problems

def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def evaluate_llama(
    ckpt_dir: str,
    tokenizer_path: str,
    result_dir: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
    num_samples_per_task: int = 1,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )

    problems = read_problems()

    # Currently sampling only once for each input
    num_tasks = len(problems.keys())
    completions = []
    task_ids = list(problems.keys())[:num_tasks]
    prompts = [problems[task_id]["prompt"] for task_id in task_ids]

    for _ in range(num_samples_per_task):
        cur_completions = []
        for i in range(num_tasks//max_batch_size+1):
            cur_completions += generator.generate(prompts[i*max_batch_size:(i+1)*max_batch_size], max_gen_len=256, temperature=temperature, top_p=top_p)
        completions.append(cur_completions)

    samples = [
        dict(task_id=task_id, completion=completions[sample_num][i])
        for i, task_id in enumerate(task_ids)
        for sample_num in range(num_samples_per_task)
        ]

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    write_jsonl(result_dir+"samples.jsonl", samples)
    results = evaluate_functional_correctness(result_dir+"samples.jsonl")
    
    # write results to file
    print(results)
    with open(result_dir+'results.txt', 'w') as f:
        f.write(str(results))

def main(
        model_size='13B',
        num_samples_per_task=1,
        local_rank: int = 0,
        ):
    
    llama_dir = '/share/models/llama_model/llama/'
    ckpt_dir = llama_dir+model_size
    tokenizer_path = llama_dir+'tokenizer.model'
    result_dir = 'results/' + model_size + '_' + str(num_samples_per_task) + '/'

    evaluate_llama(ckpt_dir=ckpt_dir, tokenizer_path=tokenizer_path, result_dir=result_dir, num_samples_per_task=num_samples_per_task)

if __name__ == "__main__":
    fire.Fire(main)
