from lmw.demo_watermark import load_model, generate, detect, parse_args, load_tokenizer_device
from program_perturb import perturb
from human_eval.data import write_jsonl, read_problems
import json
import os
from run_watermark import get_value_from_tuple_list
from human_eval.evaluation import evaluate_functional_correctness
from pprint import pprint
import re


def main(args, result_dir):
    args.normalizers = (args.normalizers.split(",") if args.normalizers else [])

    with open(result_dir + "original/watermarked_samples.jsonl_results.jsonl", 'r') as f:
        watermarked_samples = [json.loads(line) for line in f]
    
    

    
    with open(result_dir + "original/without_watermark_samples.jsonl_results.jsonl", 'r') as f:
        nonwatermarked_samples = [json.loads(line) for line in f]


    if not args.skip_model_load:
       tokenizer, device = load_tokenizer_device(args)
    else:
       tokenizer, device = None, None
    
    
    problems = read_problems()
    num_tasks = len(problems.keys())
    task_ids = list(problems.keys())[:num_tasks]
    term_width = 80
    


    # Currently sampling only once for each input

    for perturbation_id in list(map(int, args.perturbation_ids.split())):
        true_positive, false_positive = 0, 0
        decoded_output_with_watermark_lst, decoded_output_without_watermark_lst, with_watermark_detection_result_lst, without_watermark_detection_result_lst = [], [], [], []
        watermarked_perturbation, standard_perturbation = None, None
        
        for i, task_id in enumerate(task_ids):
            prompt = problems[task_id]["prompt"]
            args.default_prompt = prompt
            print("#"*term_width)
            print("Prompt:")
            print(prompt)

            
            try:
                watermarked_perturbation = perturb(watermarked_samples[i]['completion'], perturbation_id)[2]
                watermarked_output = watermarked_perturbation['result']
                if watermarked_output.endswith('"""\n') and watermarked_perturbation['changed'] == False:
                    watermarked_output = watermarked_samples[i]['completion']
                
            except Exception as error:
                watermarked_output = watermarked_samples[i]['completion']
     
            try:
                standard_perturbation = perturb(nonwatermarked_samples[i]['completion'], perturbation_id)[2]
                standard_output = standard_perturbation['result']
                if standard_output.endswith('"""\n') and standard_perturbation['changed'] == False:
                    standard_output = nonwatermarked_samples[i]['completion']
            
            except Exception as error:
                standard_output = nonwatermarked_samples[i]['completion']

            idx = watermarked_output.index('"""', watermarked_output.index('"""') + 1) + 5 if '"""' in watermarked_output else watermarked_output.index("'''", watermarked_output.index("'''") + 1) + 5

            with_watermark_detection_result = detect(watermarked_output[idx:-1], 
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
            

            
            idx = standard_output.index('"""', standard_output.index('"""') + 1) + 5 if '"""' in standard_output else standard_output.index("'''", standard_output.index("'''") + 1) + 5

           


            without_watermark_detection_result = detect(standard_output[idx:-1], 
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

            watermarked_samples.append(dict(task_id=task_id, completion= watermarked_output))
            nonwatermarked_samples.append(dict(task_id=task_id, completion= standard_output))

            decoded_output_without_watermark_lst.append(watermarked_output)
            decoded_output_with_watermark_lst.append(standard_output)


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
        

        
        perturbed_result_dir = result_dir + re.search(r"<function\s+(.*?)\s+at", str(watermarked_perturbation['the_seq'][0])).group(1) + '/'
        if not os.path.exists(perturbed_result_dir):
            os.makedirs(perturbed_result_dir)

        write_jsonl(perturbed_result_dir+"watermarked_samples.jsonl", watermarked_samples)
        watermarked_results = evaluate_functional_correctness(perturbed_result_dir+"watermarked_samples.jsonl")

        write_jsonl(perturbed_result_dir+"without_watermark_samples.jsonl", nonwatermarked_samples)
        without_watermark_results = evaluate_functional_correctness(perturbed_result_dir+"without_watermark_samples.jsonl")
        
        # write results to file
        print('watermarked results:\n', watermarked_results)
        with open(perturbed_result_dir+'watermarked_results.txt', 'w') as f:
            f.write(str(watermarked_results))
        
        print('without watermark results:\n', without_watermark_results)
        with open(perturbed_result_dir+'without_watermark_results.txt', 'w') as f:
            f.write(str(without_watermark_results))
        
        print('True positives: ', true_positive, '/', len(task_ids))
        print('False positives: ', false_positive, '/', len(task_ids))

        # write results to file
        with open(perturbed_result_dir+'watermarked_detections.txt', 'w') as f:
            f.writelines(str(with_watermark_detection_result_lst))
        
        with open(perturbed_result_dir+'without_watermark_detections.txt', 'w') as f:
            f.writelines(str(without_watermark_detection_result_lst))

        with open(perturbed_result_dir+'true_positive_rate.txt', 'w') as f:
            f.write(str(true_positive/len(task_ids)))

        with open(perturbed_result_dir+'false_positive_rate.txt', 'w') as f:
            f.write(str(false_positive/len(task_ids)))
                

if __name__ == "__main__":
    args = parse_args()
    print(args)
    result_dir = f'results/watermarking/{args.model_size}/'
    main(args, result_dir)    