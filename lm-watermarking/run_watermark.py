from demo_watermark import load_model, generate, detect, parse_args, run_gradio
from human_eval.data import write_jsonl, read_problems
from human_eval.evaluation import evaluate_functional_correctness
import os
from pprint import pprint



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

    # Generate and detect, report to stdout
    if not args.skip_model_load:
        

        problems = read_problems()

        # Currently sampling only once for each input
        num_tasks = len(problems.keys())
        task_ids = list(problems.keys())[:num_tasks]
        prompts = [problems[task_id]["prompt"] for task_id in task_ids]


        decoded_output_with_watermark_lst, decoded_output_without_watermark_lst, with_watermark_detection_result_lst, without_watermark_detection_result_lst = [], [], [], []


        for prompt in prompts:

            args.default_prompt = prompt

            term_width = 80
            print("#"*term_width)
            print("Prompt:")
            print(prompt)

            _, _, decoded_output_without_watermark, decoded_output_with_watermark, _ = generate(prompt, 
                                                                                                args, 
                                                                                                model=model, 
                                                                                                device=device, 
                                                                                                tokenizer=tokenizer)
            without_watermark_detection_result = detect(decoded_output_without_watermark, 
                                                        args, 
                                                        device=device, 
                                                        tokenizer=tokenizer)
            with_watermark_detection_result = detect(decoded_output_with_watermark, 
                                                    args, 
                                                    device=device, 
                                                    tokenizer=tokenizer)
            
            decoded_output_without_watermark_lst.append(decoded_output_without_watermark)
            decoded_output_with_watermark_lst.append(decoded_output_with_watermark)
            with_watermark_detection_result_lst.append(with_watermark_detection_result)
            without_watermark_detection_result_lst.append(without_watermark_detection_result)

            print("#"*term_width)
            print("Output without watermark:")
            print(decoded_output_without_watermark)
            print("-"*term_width)
            print(f"Detection result @ {args.detection_z_threshold}:")
            pprint(without_watermark_detection_result)
            print("-"*term_width)

            print("#"*term_width)
            print("Output with watermark:")
            print(decoded_output_with_watermark)
            print("-"*term_width)
            print(f"Detection result @ {args.detection_z_threshold}:")
            pprint(with_watermark_detection_result)
            print("-"*term_width)


    # Launch the app to generate and detect interactively (implements the hf space demo)
    if args.run_gradio:
        run_gradio(args, model=model, tokenizer=tokenizer, device=device)


    watermarked_samples = [
        dict(task_id=task_id, completion= decoded_output_with_watermark_lst[sample_num][i])
        for i, task_id in enumerate(task_ids)
        for sample_num in range(num_samples_per_task)
        ]
    
    without_watermark_samples = [
        dict(task_id=task_id, completion= decoded_output_without_watermark_lst[sample_num][i])
        for i, task_id in enumerate(task_ids)
        for sample_num in range(num_samples_per_task)
        ]
    

    

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    write_jsonl(result_dir+"watermarked_samples.jsonl", watermarked_samples)
    watermarked_results = evaluate_functional_correctness(result_dir+"watermarked_samples.jsonl")



    write_jsonl(result_dir+"without_watermark_samples.jsonl", without_watermark_samples)
    without_watermark_results = evaluate_functional_correctness(result_dir+"without_watermark_samples.jsonl")



    
    # write results to file
    print('watermarked results:', watermarked_results)
    with open(result_dir+'watermarked_results.txt', 'w') as f:
        f.write(str(watermarked_results))
    

    print('without watermark results:', without_watermark_results)
    with open(result_dir+'without_watermark_results.txt', 'w') as f:
        f.write(str(without_watermark_results))
    
    # write results to file
    print('watermarked detections:', with_watermark_detection_result_lst)
    with open(result_dir+'watermarked_detections.txt', 'w') as f:
        f.writelines(with_watermark_detection_result_lst)
    

    print('without watermark detections:', without_watermark_detection_result_lst)
    with open(result_dir+'without_watermark_detections.txt', 'w') as f:
        f.writelines(without_watermark_detection_result_lst)

    return

if __name__ == "__main__":

    args = parse_args()
    print(args)
    result_dir = 'results/' + 'watermarking/' + args.model_size + '/'
    main(args, result_dir)