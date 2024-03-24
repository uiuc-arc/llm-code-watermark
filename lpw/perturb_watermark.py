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
import nltk
import scipy
import time

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
    return f"   {completion}"


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

    ids_cached = [0]

    for depth in list(map(int, args.depths.split())):
        for psi in list(map(int, args.psis.split())):
            for perturbation_id in list(map(int, args.perturbation_ids.split())):
                true_positive, false_positive = 0, 0
                decoded_output_with_watermark_lst, decoded_output_without_watermark_lst, with_watermark_detection_result_lst, without_watermark_detection_result_lst = [], [], [], []
                watermarked_perturbation, standard_perturbation = None, None
                fraction_green = []
                comments_set = set()
                prop16 = 0
                tottoks = 0
                res_lst = []
                p_vals = []
                ytrue = []
                ypred = []
                changed = []
                green_changed = []

                k = 1
                for i in range(args.num_codes):
                    print("#"*term_width)
                    watermarked_output = ""
                    standard_output = ""
                    unperturbed_output = ""
                    j = 0
                    while j < args.num_prompts:
                        if not ids_cached[0]:
                            idx = random.randrange(0, len(watermarked_samples))
                        else:
                            idx = ids_cached[k]
                        
                        try:
                            exec(watermarked_samples[idx])

                        except:
                            individual_watermarked = watermarked_samples[idx]
                            individual_watermarked = re.sub(r'#.*', '', individual_watermarked)
                            comments_set.add(get_completion(individual_watermarked))
                            
                            continue
                        try:
                            exec(nonwatermarked_samples[idx])
                        except:
                            continue

                        individual_watermarked = watermarked_samples[idx]
                        
                        
                        unperturbed_output += individual_watermarked
                        
                        # if perturbation_id:
                        #     import pdb;pdb.set_trace()
                        exec(individual_watermarked)

                        if get_completion(astor.to_source(ast.parse(individual_watermarked))) == '   ':
                            
                            continue

                        watermarked_perturbation = program_perturb_cst.perturb(individual_watermarked, perturbation_id, depth= depth, samples= 1, psi = psi)[-1]
                        


                        
                        individual_function = nonwatermarked_samples[idx]

                        if get_completion(astor.to_source(ast.parse(individual_function)))  == '   ':
                            continue

                        standard_perturbation = program_perturb_cst.perturb(individual_function, perturbation_id, depth = depth, psi = psi)[-1]
                        
                        
                        watermarked_output += watermarked_perturbation['result']
                        standard_output += standard_perturbation['result']
                        j += 1
                        if not ids_cached[0]:
                            ids_cached.append(idx)
                        k += 1

                    unperturbed_detection_result = detect(get_completion(unperturbed_output), args, device=device, tokenizer=tokenizer, unigram_detector= unigram_detector)
                    unperturbed_z_score = float(get_value_from_tuple_list(unperturbed_detection_result[0], 'z-score'))

                    try:
                        with_watermark_detection_result = detect(get_completion(watermarked_output), args, device=device, tokenizer=tokenizer, unigram_detector= unigram_detector)
                    except:
                        import pdb;pdb.set_trace()
                    
                    watermarked_z_score = float(get_value_from_tuple_list(with_watermark_detection_result[0], 'z-score'))
                    p_vals.append(scipy.stats.norm.sf(float(get_value_from_tuple_list(with_watermark_detection_result[0], 'z-score'))))


                    res_lst.append({'Total Before': get_value_from_tuple_list(unperturbed_detection_result[0], 'Tokens Counted (T)'), 'Total After':  get_value_from_tuple_list(with_watermark_detection_result[0], 'Tokens Counted (T)'), 'Green Before': get_value_from_tuple_list(unperturbed_detection_result[0], '# Tokens in Greenlist') , 'Green After':  get_value_from_tuple_list(with_watermark_detection_result[0], '# Tokens in Greenlist')})
                    changed.append(abs(float(get_value_from_tuple_list(with_watermark_detection_result[0], 'Tokens Counted (T)')) - float(get_value_from_tuple_list(unperturbed_detection_result[0], 'Tokens Counted (T)')))/(float(get_value_from_tuple_list(unperturbed_detection_result[0], 'Tokens Counted (T)'))))
                    green_changed.append(abs(float(get_value_from_tuple_list(with_watermark_detection_result[0], 'Tokens Counted (T)')) - float(get_value_from_tuple_list(unperturbed_detection_result[0], 'Tokens Counted (T)')))/(float(get_value_from_tuple_list(unperturbed_detection_result[0], 'Tokens Counted (T)'))))

                    tmp = int(float(get_value_from_tuple_list(with_watermark_detection_result[0], 'Tokens Counted (T)')))
                    if tmp < 16:
                        prop16 += 1
                    tottoks += tmp
                    fraction_green.append(float(get_value_from_tuple_list(with_watermark_detection_result[0], '# Tokens in Greenlist'))/ float(get_value_from_tuple_list(with_watermark_detection_result[0], 'Tokens Counted (T)')))

                    print('Watermarked_z_score:', watermarked_z_score)
                    ytrue.append(1)
                    if watermarked_z_score > args.detection_z_threshold:
                        true_positive += 1
                        print("True positive!")
                        ypred.append(1)
                    else:
                        print("False negative!")
                        ypred.append(0)
                    
                    try:
                        without_watermark_detection_result = detect(get_completion(standard_output), 
                                                                    args, 
                                                                    device=device, 
                                                                    tokenizer=tokenizer, unigram_detector= unigram_detector)
                    except:
                        import pdb;pdb.set_trace()
                    
                    nonwatermarked_z_score = float(get_value_from_tuple_list(without_watermark_detection_result[0], 'z-score'))

                    print('Nonwatermarked_z_score:', nonwatermarked_z_score)
                    ytrue.append(0)
                    if nonwatermarked_z_score > args.detection_z_threshold:
                        false_positive += 1
                        print("False positive!")
                        ypred.append(1)
                    else:
                        print("True negative!")
                        ypred.append(0)

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

                ids_cached[0] = 1
                perturbed_result_dir = result_dir + str(args.num_prompts) + '/' + re.search(r"<function\s+(.*?)\s+at", str(watermarked_perturbation['the_seq'][0])).group(1) + '/' + str(depth) + '/' + str(psi) + '/'
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

                with open(perturbed_result_dir+'detection_results.txt', 'w') as f:
                    f.write(str({'tpr': round(true_positive/args.num_codes, 3), 'fpr': round(false_positive/args.num_codes, 3), 'p-values': p_vals, 'ytrue': ytrue, 'ypred': ypred, 'mean_tokens': tottoks/args.num_codes, 'prop_tokens_changed': sum(changed)/args.num_codes, 'prop_green_changed': sum(green_changed)/args.num_codes}))
                
                with open(perturbed_result_dir+'res.txt', 'w') as f:
                    for dct in res_lst:
                        f.write(str(dct) + '\n')

                with open(perturbed_result_dir+'number of skipped', 'w') as f:
                    f.write(str(len(comments_set)))

if __name__ == "__main__":
    args = parse_args()
    if args.download_wordnet:
        nltk.download('wordnet')
    sz = args.model_size if not args.use_codellama else f'CodeLlama{args.model_size}'
    if args.use_robdist:
        result_dir = f'results/watermarking/{sz}/{args.dataset}/{args.language}/{args.gamma}/robdist/'
    elif args.sweet_threshold:
        result_dir = f'results/watermarking/{sz}/{args.dataset}/{args.language}/{args.gamma}/sweet/'
    elif args.use_unigram:
        result_dir = f'results/watermarking/{sz}/{args.dataset}/{args.language}/{args.gamma}/unigram/'
    else:
        result_dir = f'results/watermarking/{sz}/{args.dataset}/{args.language}//{args.gamma}/vanilla/'
    main(args, result_dir)    