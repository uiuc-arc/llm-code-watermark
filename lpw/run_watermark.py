import torch
import os

from lmw.demo_watermark import load_model, generate, detect, parse_args
from human_eval.data import write_jsonl, read_problems
from human_eval.evaluation import evaluate_functional_correctness
from pprint import pprint
from fairscale.nn.model_parallel.initialize import initialize_model_parallel


# reference: https://github.com/declare-lab/instruct-eval/blob/main/human_eval/main.py#L35
def filter_code(completion: str) -> str:
    # The program tends to overwrite, we only take the first function
    completion = completion.lstrip("\n")
    return completion.split("\n\n")[0]


def fix_indents(text: str) -> str:
    return text.replace("\t", "    ")


def get_value_from_tuple_list(lst, key):
    for item in lst:
        if item[0] == key:
            return item[1]
    return 0

def main(args, result_dir, num_samples_per_task = 1): 
    """Run a command line version of the generation and detection operations
        and optionally launch and serve the gradio demo"""
    
    # Initial arg processing and log
    args.normalizers = (args.normalizers.split(",") if args.normalizers else [])
    print(args)

    if not args.skip_model_load:
        model, tokenizer, device = load_model(args)
    else:
        model, tokenizer, device = None, None, None
    
    watermarked_samples = []
    nonwatermarked_samples = []

    # Generate and detect, report to stdout
    if not args.skip_model_load:
        problems = read_problems()

        # Currently sampling only once for each input
        num_tasks = len(problems.keys())
        task_ids = list(problems.keys())[:num_tasks]

        decoded_output_with_watermark_lst, decoded_output_without_watermark_lst, with_watermark_detection_result_lst, without_watermark_detection_result_lst = [], [], [], []
        watermarked_completions, standard_completions = [], []
        true_positive, false_positive = 0, 0

        for task_id in task_ids:
            prompt = problems[task_id]["prompt"]
            args.default_prompt = prompt
            term_width = 80
            print("#"*term_width)
            print("Prompt:")
            print(prompt)

            _, _, standard_output, watermarked_output, _ = generate(prompt, args, model=model, device=device, tokenizer=tokenizer)
            
            watermarked_output= filter_code(fix_indents(watermarked_output))
            standard_output= filter_code(fix_indents(standard_output))
            watermarked_completions.append(watermarked_output)
            standard_completions.append(standard_output)

            # detect with watermark
            with_watermark_detection_result = detect(watermarked_output, 
                                                    args, 
                                                    device=device, 
                                                    tokenizer=tokenizer)
            
            watermarked_z_score = float(get_value_from_tuple_list(with_watermark_detection_result[0], 'z-score'))
            print('Watermarked_z_score:', watermarked_z_score)
            if watermarked_z_score > args.detection_z_threshold:
                true_positive += 1
                print("True positive!")
            else:
                print("False negative!")
            
            # detect without watermark
            without_watermark_detection_result = detect(standard_output, 
                                                        args, 
                                                        device=device, 
                                                        tokenizer=tokenizer)
            
            nonwatermarked_z_score = float(get_value_from_tuple_list(without_watermark_detection_result[0], 'z-score'))
            print('Nonwatermarked_z_score:', nonwatermarked_z_score)
            if nonwatermarked_z_score > args.detection_z_threshold:
                false_positive += 1
                print("False positive!")
            else:
                print("True negative!")

            watermarked_samples.append(dict(task_id=task_id, completion=f"{prompt} {watermarked_output}"))
            nonwatermarked_samples.append(dict(task_id=task_id, completion=f"{prompt} {standard_output}"))

            decoded_output_without_watermark_lst.append(f"{prompt} {watermarked_output}")
            decoded_output_with_watermark_lst.append(f"{prompt} {standard_output}")
            with_watermark_detection_result_lst.append(with_watermark_detection_result)
            without_watermark_detection_result_lst.append(without_watermark_detection_result)

            print("#"*term_width)
            print("Output without watermark:")
            print(watermarked_output)
            print("-"*term_width)
            print(f"Detection result @ {args.detection_z_threshold}:")
            pprint(without_watermark_detection_result)
            print("-"*term_width)

            print("#"*term_width)
            print("Output with watermark:")
            print(standard_output)
            print("-"*term_width)
            print(f"Detection result @ {args.detection_z_threshold}:")
            pprint(with_watermark_detection_result)
            print("-"*term_width)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    write_jsonl(result_dir+"watermarked_samples.jsonl", watermarked_samples)
    watermarked_results = evaluate_functional_correctness(result_dir+"watermarked_samples.jsonl")

    write_jsonl(result_dir+"without_watermark_samples.jsonl", nonwatermarked_samples)
    without_watermark_results = evaluate_functional_correctness(result_dir+"without_watermark_samples.jsonl")
    
    # write results to file
    print('watermarked results:\n', watermarked_results)
    with open(result_dir+'watermarked_results.txt', 'w') as f:
        f.write(str(watermarked_results))
    
    print('without watermark results:\n', without_watermark_results)
    with open(result_dir+'without_watermark_results.txt', 'w') as f:
        f.write(str(without_watermark_results))
    
    print('True positives: ', true_positive, '/', len(task_ids))
    print('False positives: ', false_positive, '/', len(task_ids))

    # write results to file
    with open(result_dir+'watermarked_detections.txt', 'w') as f:
        f.writelines(str(with_watermark_detection_result_lst))
    
    with open(result_dir+'without_watermark_detections.txt', 'w') as f:
        f.writelines(str(without_watermark_detection_result_lst))

    with open(result_dir+'true_positive_rate.txt', 'w') as f:
        f.write(str(true_positive/len(task_ids)))

    with open(result_dir+'false_positive_rate.txt', 'w') as f:
        f.write(str(false_positive/len(task_ids)))
    
    with open(result_dir + 'watermarked_completions.txt', 'w') as f:
        f.writelines(str(watermarked_completions))

    with open(result_dir + 'standard_completions.txt', 'w') as f:
        f.writelines(str(standard_completions))

    return

if __name__ == "__main__":
    args = parse_args()
    print(args)
    result_dir = f'results/watermarking/{args.model_size}/'

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    main(args, result_dir)
