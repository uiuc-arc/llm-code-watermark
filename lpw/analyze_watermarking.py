import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from lmw.demo_watermark import parse_args
import ast



def get_results_dataframe(result_dir):
    df = pd.DataFrame(columns = ['perturbation', 'tpr', 'fpr', 'watermarked_pass@1', 'standard_pass@1'])
    
    watermarked_pass_k = []
    standard_pass_k = []
    
    for sub in os.scandir(result_dir):
        if sub.is_dir():
            with open(f'{sub.path}/true_positive_rate.txt', 'r') as f:
                tpr = float(f.readlines()[0])
            
            with open(f'{sub.path}/false_positive_rate.txt', 'r') as f:
                fpr = float(f.readlines()[0])
        
            
            with open(f'{sub.path}/watermarked_results.txt', 'r') as f:
                for line in f:
                    watermarked_pass_k.append(ast.literal_eval(line))
                    
            
            with open(f'{sub.path}/without_watermark_results.txt', 'r') as f:
                for line in f:
                    standard_pass_k.append(ast.literal_eval(line))
            
            df = df._append({'perturbation': sub.name, 'tpr': tpr, 'fpr': fpr, 'watermarked_pass@1': watermarked_pass_k[0]['pass@1'], 'standard_pass@1': standard_pass_k[0]['pass@1']}, ignore_index = True)
    
    print(df)
    return df


def plot_tpr_fpr(df, result_dir):

    plot = df.plot(x="perturbation", y=["tpr", "fpr"], kind="bar")

    plot.tick_params(axis='x', labelsize=8)  

    plt.title('TPR and FPR Rates v Perturbation')

    plt.savefig(f'{result_dir}/tpr_fpr_plot.png')
    plt.close('all')


if __name__ == "__main__":
    args = parse_args()
    print(args)
    result_dir = f'results/watermarking/{args.model_size}/'
    df = get_results_dataframe(result_dir)
    plot_tpr_fpr(df, result_dir)



