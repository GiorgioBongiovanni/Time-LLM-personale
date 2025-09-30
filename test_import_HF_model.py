# Test 1
from huggingface_hub import HfApi
api = HfApi()
api.model_info("meta-llama/Llama-3.2-3B")
print("Access OK")

# Test 2
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("meta-llama/Llama-3.2-3B")
print(cfg.hidden_size)