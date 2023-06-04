Install human-eval: https://github.com/openai/human-eval

Install llama: https://github.com/facebookresearch/llama

Model checkpoint location is hardcoded to: `/share/models/llama_model/llama/` for now 

Run following command 
```
python3 -m torch.distributed.launch --nproc_per_node=1 lpw/run_llama.py
```
