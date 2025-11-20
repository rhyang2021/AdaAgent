import os
import traceback
import pdb
import json
import sys
from openai import OpenAI, AzureOpenAI
from hunyuan_api import Api
import requests
from requests.exceptions import Timeout
from typing import Dict, List
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
sys.path.append("/apdcephfs_cq11/share_1567347/share_info/rhyang/NLGames/src")
from call_llm_apis import send_request, send_conversation_request
  
# @retry(wait=wait_random_exponential(min=5, max=10), stop=stop_after_attempt(20))
def llm_openai(prompt: List[Dict[str, str]], model: str):
    """Get completion from the GPT model."""
    client = OpenAI(base_url="https://gptproxy.llmpaas.woa.com/v1", 
                    api_key=os.getenv("OPENAI_API_KEY"))
    for i in range(5):
        try:
            response = client.chat.completions.create(
                model=model, 
                # messages=[
                # {"role": "system", "content": "You are a helpful assistant."},
                # {"role": "user", "content": user_prompt},
                # ],
                messages = prompt,
                temperature=1,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f'ERROR: {str(e)}')
            print(f'Retrying ({i + 1}/5), wait for {2 ** (i + 1)} sec...')
            time.sleep(2 ** (i + 1))
    return None

def llm_deepseek(prompt: List[Dict[str, str]], model: str):
    """Get completion from the GPT model."""
    client = OpenAI(base_url="https://api.deepseek.com", 
                    api_key=os.getenv("DEEPSEEK_API_KEY"))
    for i in range(5):
        try:
            response = client.chat.completions.create(
                model=model, 
                # messages=[
                # {"role": "system", "content": "You are a helpful assistant."},
                # {"role": "user", "content": user_prompt},
                # ],
                messages = prompt,
                temperature=0,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f'ERROR: {str(e)}')
            print(f'Retrying ({i + 1}/5), wait for {2 ** (i + 1)} sec...')
            time.sleep(2 ** (i + 1))
    return None

@retry(wait=wait_random_exponential(min=5, max=10), stop=stop_after_attempt(2))
def llm_gpt(prompt: List[Dict[str, str]], model: str):
    """Get completion from the GPT model."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    for i in range(5):
        try:
            response = client.chat.completions.create(
                model=model, 
                messages = prompt,
                temperature=1,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f'ERROR: {str(e)}')
            print(f'Retrying ({i + 1}/5), wait for {2 ** (i + 1)} sec...')
            time.sleep(2 ** (i + 1))
    return None


# @retry(wait=wait_random_exponential(min=5, max=10), stop=stop_after_attempt(2))
def llm_azure(prompt: List[Dict[str, str]], temperature=1):
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"), 
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
        api_version="2024-02-01"
        )
    for i in range(5):
        try:
            response = client.chat.completions.create(
                model="gpt-4o", # model = "deployment_name".
                messages=prompt,
                temperature=temperature
                )
            return response.choices[0].message.content
        except Exception as e:
            print(f'ERROR: {str(e)}')
            print(f'Retrying ({i + 1}/5), wait for {2 ** (i + 1)} sec...')
            time.sleep(2 ** (i + 1))
    return None

@retry(wait=wait_random_exponential(min=5, max=10), stop=stop_after_attempt(2))
def llm_gpt_mini(prompt: List[Dict[str, str]], temperature=0):
    client = AzureOpenAI(
        azure_endpoint="https://eastus.api.cognitive.microsoft.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-08-01-preview", 
        api_key="2b1125d69663443aaaa39a4713c6dcab",  
        api_version="2024-02-01"
        )
    for i in range(5):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini", # model = "deployment_name".
                messages=prompt,
                temperature=temperature
                )
            return response.choices[0].message.content
        except Exception as e:
            print(f'ERROR: {str(e)}')
            print(f'Retrying ({i + 1}/5), wait for {2 ** (i + 1)} sec...')
            time.sleep(2 ** (i + 1))
    return None

    
def llm_hunyuan(messages: List[Dict]) -> str:
    """腾讯混元调用"""
    for i in range(5):
        try:
            from hunyuan_api import Api
            HOST = "trpc-gpt-eval.production.polaris"  # 生产
            api = Api("http://{}:8080".format(HOST), "BcsdPalE_fanghuaye", "ycZSTuDOGb48CiNh")
            user_msg = []
            for item in messages:
                user_msg.append({
                    "role": item["role"],
                    "content": [{"type": "text", "value": item["content"]}]
                })
            ret_msg = api.chat(user_msg)
            content = ret_msg.json()['answer'][0]['value']
            
            return content
        except Exception as e:
            print(f'ERROR: {str(e)}')
            print(f'Retrying ({i + 1}/5), wait for {2 ** (i + 1)} sec...')
            time.sleep(2 ** (i + 1))
    return "Hunyuan API Error"

def llm_claude(prompt: List[Dict[str, str]], temperature=0):
    api_key = "ruotianma_key"
    temperature_value = 0
    conversation_history = []
    for item in prompt:
        if item["role"] == "system" or item["role"] == "user":
            conversation_history.append(
                {"role": "user",
                "parts": {"text": item["content"]}}
            )
        else:
            conversation_history.append(
                {"role": "model",
                "parts": {"text": item["content"]}
                })
    result_multi = send_conversation_request(api_key, conversation_history, 'claude-3-5-sonnet@20240620', temperature=temperature)

    try:
        data = json.loads(result_multi['result']) 
        return data["content"][0]["text"]
    except:
        return ""

def vllm(prompt: List[Dict[str, str]], model: str, port=8401, temperature=1) -> str:

    if 'llama3.1-8b' in model:    
        if "cog-sft_sci" in model:
            model_id = "/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/models/llama3.1-8b_cog-sft_balance_sci_lr2e6_bs32_epoch3_full_1021/checkpoint-1500"
        elif "cog-sft_alf" in model:
            model_id = "/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/models/llama3.1-8b_cog-sft_balance_alf_lr2e6_bs16_epoch5_full_1026"
        elif 'rlvcr_alf' in model:
            model_id = "/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/models/llama3.1_alf_cold_start_rlvcr_verl_v2_kl0.1_1030_ckp130"
        elif 'gigpo_sci' in model:
            model_id = "/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/models/llama3.1-8b_sci_cold_start_gigpo_kl0.05_1108_ckp130"
        elif 'gigpo_alf' in model:
            model_id = "/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/models/llama3.1-8b_alf_cold_start_gigpo_kl0.01_1106_ckp140"
        elif 'rlvcr_sci' in model:
            model_id = "/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/models/llama3.1_sci_cold_start_rlvcr_verl_v2_kl0.2_1027_ckp30"
        elif 'distill' in model:
            model_id = "/apdcephfs_cq11/share_1567347/share_info/llm_models/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/6a6f4aa4197940add57724a7707d069478df56b1"
        elif "sft_alf" in model:
            model_id = "/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/models/llama3.1-8b_sft_alf_lr2e6_bs32_epoch5_full_1021"
        elif "sft_sci" in model:
            model_id = "/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/models/llama3.1-8b_sft_sci_lr2e6_bs32_epoch5_full_1021"
        elif "sft" in model:
            model_id = "/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/models/llama3-8b_sft_sci"
        else:
            model_id = "/apdcephfs_cq11/share_1567347/share_info/llm_models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
        base_url=f"http://29.119.98.176:{port}/v1"
    if "QwQ-" in model:
        model_id = "/apdcephfs_cq11/share_1567347/share_info/llm_models/models--Qwen--QwQ-32B/snapshots/976055f8c83f394f35dbd3ab09a285a984907bd0"
        base_url=f"http://11.216.48.78:{port}/v1"
    elif 'qwen2.5-7b' in model:
        if 'grpo_alf' in model:
            model_id = "/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/models/qwen2.5-7b_alf_cold_start_grpo_verl_v2_kl0.01_0915_ckp140"
        elif 'grpo_sci' in model:
            model_id = "/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/models/qwen2.5-7b_sci_cold_start_grpo_verl_v2_kl0.05_1023_ckp130"
        elif 'gigpo_alf' in model:
            model_id = "/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/models/qwen2.5-7b_alf_cold_start_gigpo_kl0.01_1101_ckp40"
        elif 'gigpo_sci' in model:
            model_id = "/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/models/qwen2.5-7b_sci_cold_start_gigpo_verl_v2_kl0.05_1103_ckp40"
        elif 'distill' in model:
            model_id = "/apdcephfs_cq11/share_1567347/share_info/llm_models/DeepSeek-R1-Distill-Qwen-7B"
        elif 'rlvcr_alf' in model:
            model_id = "/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/models/qwen2.5-7b_alf_cold_start_rlvcr_verl_v2_kl0.1_0923_ckp140"
        elif 'rlvcr_sci' in model:
            model_id = "/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/models/qwen2.5-7b_sci_cold_start_rlvcr_verl_v2_kl0.2_1019_ckp30"
        elif 'cog-sft_alf' in model:
            model_id = "/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/models/qwen2.5-7b_cog-sft_alf_lr2e6_bs16_epoch5_full_0810"
        elif 'cog-sft_balance_alf' in model:
            model_id = "/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/models/qwen2.5-7b_cog-sft_balance_alf_lr2e6_bs16_epoch5_full_0915"
        elif 'cog-sft_balance_sci_verl' in model:
            model_id = "/apdcephfs_cq11_1567347/share_1567347/share_info/rhyang/AdaAgent/models/qwen2.5-7b_cog-sft_verl_sci_lr2e6_bs32_epoch3_full_1018"
        elif 'cog-sft_sci' in model:
            model_id = "/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/models/qwen2.5-7b_cog-sft_sci_lr2e6_bs16_epoch5_full_0810"
        elif 'cog-sft_balance_sci' in model:
            model_id = "/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/models/qwen2.5-7b_cog-sft_balance_sci_lr2e6_bs16_epoch5_full_0915"
        elif 'sft_alf' in model:
            model_id = "/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/models/qwen2.5-7b_sft_alf_lr2e6_bs8_epoch5_full_0722"
        elif 'sft_sci' in model:
            model_id = "/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/models/qwen2.5-7b_sft_sci_lr2e6_bs8_epoch5_full_0625/checkpoint-600"
        else:
            model_id = "/apdcephfs_cq11/share_1567347/share_info/llm_models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
        base_url=f"http://29.119.98.176:{port}/v1"
    
    elif "r1-" in model:
        model_id = "/apdcephfs_cq11/share_1567347/share_info/llm_models/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/6a6f4aa4197940add57724a7707d069478df56b1"
        base_url=f"http://29.232.241.115:{port}/v1"
    
    client = OpenAI(
            base_url=base_url,
            api_key="EMPTY"
            )
    
    for i in range(5):
        try:
            completion = client.chat.completions.create(
                model=model_id,
                messages=prompt,
                temperature=temperature,
                max_tokens=4096,
                )
            text = completion.choices[0].message.content
            return text.strip()
        
        except Exception as e:
            print(f'ERROR: {str(e)}')
            print(f'Retrying ({i + 1}/5), wait for {2 ** (i + 1)} sec...')
            time.sleep(2 ** (i + 1))
    return ""

def hf_request(prompt, port, temperature):
    url = f'http://29.79.9.233:{port}/generate'
    # request
    data = {
        'index': 1,
        'messages': prompt,
        'temperature': temperature
        }
    # POST
    response = requests.post(url, json=data)
    print(response.json())
    
    return response.json()['response']

def hf_request_3(prompt, port, temperature):
    url = f'http://29.81.244.73:{port}/generate'
    # request
    data = {
        'index': 1,
        'messages': prompt,
        'temperature': temperature
        }
    # POST
    response = requests.post(url, json=data)
    print(response.json())
    
    return response.json()['response']

if __name__ == "__main__":
    # Example usage
    prompt = [
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    response = llm_hunyuan(prompt)
    print(response)

