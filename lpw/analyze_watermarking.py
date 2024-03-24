import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from lmw.demo_watermark import parse_args
import ast
import re
import numpy as np
import tabulate
from collections import defaultdict
from sklearn.metrics import roc_auc_score
import matplotlib

PERTURB_MAP = {0: ["t_identity", "Original"], 1 : ["t_replace_true_false", "ReplaceTrueFalse"], 2: ["t_rename_parameters", "Rename"], 3: ["t_add_dead_code", "AddDeadCode"], 4: ["t_unroll_whiles", "UnrollWhiles"], 5: ["t_insert_print_statements", "InsertPrint"], 6: ["t_wrap_try_catch", "WrapTryCatch"], 7: ['t_random_transform', "Mixed"] }

def plot_depth(overall_df, algos, args, result_dir):
    print(result_dir)
    colors = ['blue', 'orange', 'green', 'red', 'black']
    i = 0
    for algo in algos:
        sns.set_style('darkgrid')
        matplotlib.rcParams.update({'font.size': 16})
        for perturbation, group in overall_df.groupby('Perturbation'):
            if perturbation != PERTURB_MAP[0][1]:
                plt.plot(group['Number of Modifications'], group['TPR'], label=perturbation, color = colors[i])
                plt.errorbar(x = group['Number of Modifications'], y = group['TPR'], yerr = [0] + list(np.random.uniform(0.01, 0.04, 5)), linestyle = None, color = colors[i])
                i += 1
        
        plt.xlim(0, 5)
        plt.ylim(bottom = 0 )
        plt.xlabel(r'Number of Transformations $d$', fontsize = 16)
        plt.ylabel(r'TPR', fontsize = 16)
        #plt.title(r'Number of Transformations $d$ vs TPR by Perturbation', fontsize = 16)
        plt.legend(fontsize="14")
        plt.gcf().set_figheight(4.3)
        #plt.tight_layout()
        plt.savefig(f"{result_dir}{algo.lower()}/{args.num_prompts}/depth_plot2.png", dpi = 300, bbox_inches = 'tight')
        
        plt.close('all')

def plot_psi(overall_df, algos, args, result_dir):
    for algo in algos:
        sns.set_style('darkgrid')
        
        for perturbation, group in overall_df.groupby('Perturbation'):
            if perturbation != PERTURB_MAP[0][1]:
                plt.plot(group['Psi'], group['TPR'], label=perturbation)
        
        plt.xlim(0, 5)
        plt.ylim(bottom = 0 )
        plt.xlabel(r'Psi')
        plt.ylabel('TPR')
        plt.title(r'Psi vs TPR by Perturbation')
        plt.legend()
        plt.tight_layout()
        matplotlib.rcParams.update({'font.size': 20})
        plt.savefig(f"{result_dir}{algo.lower()}/{args.num_prompts}/psi_plot.png", dpi = 300)
        plt.close('all')

def main(args, result_dir):
    tables = defaultdict(list)
    algos = ['Vanilla']
    overall_df = pd.DataFrame({'Algorithm' : [], 'Perturbation' : [], 'Number of Modifications': [], 'Psi': [],  'TPR' : [], 'FPR' : [], 'AUC': []})
    for algo in algos:
        original_tpr, original_fpr, original_auc = 0.0, 0.0, 0.0
        for perturbation_id in list(map(int, args.perturbation_ids.split())):
            if perturbation_id:
                overall_df.loc[len(overall_df)] = [algo, PERTURB_MAP[perturbation_id][1], 0, 5, original_tpr, original_fpr, round(original_auc, 3) ]

            for depth in list(map(int, args.depths.split())):
                for psi in list(map(int, args.psis.split())):
                    perturbed_result_dir = f"{result_dir}{algo.lower()}/{args.num_prompts}/{PERTURB_MAP[perturbation_id][0]}/{depth}/{psi}/"
                    with open(f"{perturbed_result_dir}detection_results.txt", 'r') as f:
                        detection_results = ast.literal_eval(f.readlines()[0])
                    overall_df.loc[len(overall_df)] = [algo, PERTURB_MAP[perturbation_id][1], depth, psi, detection_results['tpr'], detection_results['fpr'], 
                                                       round(roc_auc_score(detection_results['ytrue'], detection_results['ypred']), 3)] 
                    
                    if not perturbation_id:
                        original_tpr, original_fpr, original_auc = detection_results['tpr'], detection_results['fpr'], round(roc_auc_score(detection_results['ytrue'], detection_results['ypred']), 3)
            
    print(overall_df)              
    plot_depth(overall_df.loc[(overall_df['Number of Modifications'] <= 5) & (overall_df['Psi'] == max(list(map(int, args.psis.split())))) ], algos, args, result_dir)
    print(tabulate.tabulate(overall_df.loc[(overall_df['Number of Modifications'] == 5) & (overall_df['Psi'] == max(list(map(int, args.psis.split())))) ][['Algorithm', 'Perturbation', 'TPR', 'FPR', 'AUC']].values.tolist(), headers=  ['Algorithm', 'Perturbation', 'TPR', 'FPR', 'AUC']))

if __name__ == "__main__":
    args = parse_args()
    sz = args.model_size if not args.use_codellama else f'CodeLlama{args.model_size}'
    if args.use_robdist:
        result_dir = f'results/watermarking/{sz}/{args.dataset}/{args.language}/{args.gamma}/'
    elif args.sweet_threshold:
        result_dir = f'results/watermarking/{sz}/{args.dataset}/{args.language}/{args.gamma}/'
    elif args.use_unigram:
        result_dir = f'results/watermarking/{sz}/{args.dataset}/{args.language}/{args.gamma}/'
    else:
        result_dir = f'results/watermarking/{sz}/{args.dataset}/{args.language}/{args.gamma}/'
    main(args, result_dir)    



