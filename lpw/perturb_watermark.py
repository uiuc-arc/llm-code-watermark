from lmw.demo_watermark import load_model, generate, detect, parse_args, load_tokenizer_device
import program_perturb_cst
from lmw.watermark_processor import GPTWatermarkDetector
from mxeval.data import read_problems, stream_jsonl, write_jsonl, get_metadata, get_data
from mxeval.evaluation import evaluate_functional_correctness
import json
import os
from run_watermark import get_value_from_tuple_list, get_datafile
from pprint import pprint
import re
import libcst as cst
import ast
import astor
import random

def get_individual_function_lst(input_string):
    code_list = re.split(r'\n(?=def\s)', input_string)
    new_code_list = []
    imports = ""
    for code in code_list[1:]:
        segments = re.split(r'^(import .*|from .* import .*)$', code, flags=re.MULTILINE)
        new_code_list.append(imports + '\n' + segments[0])
        if len(segments) > 1:
            imports = " ".join(segments[1:])
        else:
            imports = ""
    
    return new_code_list

def get_completion(output):
        #Remove function definition
    completion = re.sub(r'\ndef \w+\(.*?\).*?:', '\n', output, flags=re.DOTALL)
    # Remove docstrings
    completion = re.sub(r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')', '',completion, flags=re.DOTALL)
    completion = re.sub(r"^\s*import\s+.+?$|^\s*from\s+.+?\s+import\s+.+?$", "", completion, flags=re.MULTILINE)
    # Remove empty lines
    completion = "\n".join([line for line in completion.splitlines() if line.strip()])
    completion = completion.strip()
    return completion



def main(args, result_dir):
    args.normalizers = (args.normalizers.split(",") if args.normalizers else [])

    with open(result_dir + "original/watermarked_decoded.txt", 'r') as f:
        watermarked_samples = ast.literal_eval(f.readlines()[0])
       
    with open(result_dir + "original/standard_decoded.txt", 'r') as f:
        nonwatermarked_samples =  ast.literal_eval(f.readlines()[0])
    
   #watermarked_samples, nonwatermarked_samples = nonwatermarked_samples, watermarked_samples #ignore this 

    if not args.skip_model_load:
       tokenizer, device = load_tokenizer_device(args)
    else:
       tokenizer, device = None, None

    term_width = 80

    unigram_detector = None if not args.use_unigram else GPTWatermarkDetector(fraction=args.unigram_fraction, strength=args.unigram_strength, vocab_size= tokenizer.vocab_size, watermark_key=args.unigram_wm_key)

    for perturbation_id in list(map(int, args.perturbation_ids.split())):
        true_positive, false_positive = 0, 0
        decoded_output_with_watermark_lst, decoded_output_without_watermark_lst, with_watermark_detection_result_lst, without_watermark_detection_result_lst = [], [], [], []
        watermarked_perturbation, standard_perturbation = None, None
        fraction_green = []

        prop16 = 0
        tottoks = 0
        
        for i in range(args.num_codes):
            print("#"*term_width)
            watermarked_output = ""
            standard_output = ""
            j = 0
            while j < args.num_prompts:
                idx = random.randrange(0, len(watermarked_samples))
                try:
                    exec(watermarked_samples[idx])
                    exec(nonwatermarked_samples[idx])

                except:
                    continue
                individual_function = watermarked_samples[idx]
                individual_function = re.sub(r'#.*', '', individual_function)
                if get_completion(individual_function) == "":
                    continue
                watermarked_perturbation = program_perturb_cst.perturb(individual_function, perturbation_id, depth= 3, samples= 1)[-1]

                individual_function = nonwatermarked_samples[i]
                individual_function = re.sub(r'#.*', '', individual_function)
                
                if get_completion(individual_function) == "":
                    continue
                
                standard_perturbation = program_perturb_cst.perturb(individual_function, perturbation_id, depth = 3)[-1]
                watermarked_output += watermarked_perturbation['result']
                standard_output += standard_perturbation['result']
                j += 1


            with_watermark_detection_result = detect(get_completion(watermarked_output), args, device=device, tokenizer=tokenizer, unigram_detector= unigram_detector)
            watermarked_z_score = float(get_value_from_tuple_list(with_watermark_detection_result[0], 'z-score'))

            tmp = int(get_value_from_tuple_list(with_watermark_detection_result[0], 'Tokens Counted (T)'))
            if tmp < 16:
                prop16 += 1
            tottoks += tmp
            fraction_green.append(float(get_value_from_tuple_list(with_watermark_detection_result[0], '# Tokens in Greenlist'))/ float(get_value_from_tuple_list(with_watermark_detection_result[0], 'Tokens Counted (T)')))

            print('Watermarked_z_score:', watermarked_z_score)
            if watermarked_z_score > args.detection_z_threshold:
                true_positive += 1
                print("True positive!")
            else:
                print("False negative!")

            without_watermark_detection_result = detect(get_completion(standard_output), 
                                                        args, 
                                                        device=device, 
                                                        tokenizer=tokenizer, unigram_detector= unigram_detector)
            
            nonwatermarked_z_score = float(get_value_from_tuple_list(without_watermark_detection_result[0], 'z-score'))

            print('Nonwatermarked_z_score:', nonwatermarked_z_score)
            if nonwatermarked_z_score > args.detection_z_threshold:
                false_positive += 1
                print("False positive!")
            else:
                print("True negative!")

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

        perturbed_result_dir = result_dir + str(args.num_prompts) + '/' + re.search(r"<function\s+(.*?)\s+at", str(watermarked_perturbation['the_seq'][0])).group(1) + '/'
        if not os.path.exists(perturbed_result_dir):
            os.makedirs(perturbed_result_dir)


        print('True positives: ', true_positive, '/', args.num_codes)
        print('False positives: ', false_positive, '/', args.num_codes)


        print('Mean Proportion in greenlist: ', sum(fraction_green)/args.num_codes)


        print('prop tokens < 16', prop16/args.num_codes)

        print('mean tokens ', tottoks/args.num_codes)

        # write results to file
        with open(perturbed_result_dir+'watermarked_detections.txt', 'w') as f:
            f.writelines(str(with_watermark_detection_result_lst))
        
        with open(perturbed_result_dir+'without_watermark_detections.txt', 'w') as f:
            f.writelines(str(without_watermark_detection_result_lst))

        with open(perturbed_result_dir+'true_positive_rate.txt', 'w') as f:
            f.write(str(true_positive/args.num_codes))

        with open(perturbed_result_dir+'false_positive_rate.txt', 'w') as f:
            f.write(str(false_positive/args.num_codes))

if __name__ == "__main__":
    args = parse_args()
    print(args)
    sz = args.model_size if not args.use_codellama else f'CodeLlama{args.model_size}'
    if args.use_robdist:
        result_dir = f'results/watermarking/{sz}/{args.dataset}/{args.language}/robdist/'
    elif args.sweet_threshold:
        result_dir = f'results/watermarking/{sz}/{args.dataset}/{args.language}/sweet/'
    elif args.use_unigram:
        result_dir = f'results/watermarking/{sz}/{args.dataset}/{args.language}/unigram/'
    else:
        result_dir = f'results/watermarking/{sz}/{args.dataset}/{args.language}/vanilla/'

    main(args, result_dir)    