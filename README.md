Install human-eval: https://github.com/openai/human-eval

Install llama: https://github.com/facebookresearch/llama

Model checkpoint location is hardcoded to: `/share/models/llama_model/llama/` for now 

Run following command 
```
python3 -m torch.distributed.launch --nproc_per_node=1 lpw/run_llama.py
```

The original LM Watermarking implementation is enabled by the [huggingface/transformers 🤗](https://github.com/huggingface/transformers) library. To convert the Llama model weights to the Hugging Face Transformers format, run the following script:

```
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /share/models/llama_model/llama/ --model_size 13B --output_dir /share/models/llama_model/hf/13B/
```

Thereafter, models can be loaded via:

```py
from transformers import LlamaForCausalLM, LlamaTokenizer

model = LlamaForCausalLM.from_pretrained("/share/models/llama_model/hf/13B/")
tokenizer = LlamaTokenizer.from_pretrained("/share/models/llama_model/hf/13B/")
```
