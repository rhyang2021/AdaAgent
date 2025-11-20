import torch
import pdb
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.stopping_criteria import StoppingCriteriaList
from safetensors import safe_open

app = Flask(__name__)

max_gpu_memory = 80
start_id = 0
num_gpus = 4
model_name = "/apdcephfs_cq11_1567347/share_1567347/share_info/llm_models/models--openai--gpt-oss-120b/snapshots/8b193b0ef83bd41b40eb71fee8f1432315e02a3e"

# kwargs = {"torch_dtype": torch.bfloat16, "offload_folder": f"{model_name}/offload"}
kwargs = {}
kwargs.update({
    "device_map": "auto",
    "max_memory": {i: f"{max_gpu_memory}GiB" for i in range(start_id, start_id + num_gpus)},
})
print(kwargs)


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # low_cpu_mem_usage=True,
    # trust_remote_code=True,
    **kwargs
    )

model.requires_grad_(False)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_name)
# if tokenizer.pad_token is None:
    # tokenizer.pad_token = tokenizer.eos_token


def llm_generation(index, messages, temperature):
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    ).to(model.device)
    
    outputs = model.generate(**inputs, max_length=8192, top_p=1.0, temperature=temperature)
    answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]).strip()

    return answer

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.json
    print(data)
    return jsonify({'response': llm_generation(data['index'], data['messages'], data['temperature'])})

if __name__ == '__main__':
    app.run(host='29.81.228.199', port=8036)