import argparse
import json
import logging
import re
import os
import time
import pdb
import ast
import requests
import sys
import builtins
import tiktoken
import numpy as np
from logging import INFO
from typing import Dict, List
from openai import OpenAI
from requests.exceptions import Timeout
from agentenv.envs import SciworldEnvClient, AlfWorldEnvClient, WebshopEnvClient, TextCraftEnvClient, WebarenaEnvClient, BabyAIEnvClient
sys.path.append("/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/utils")
sys.path.append("/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/prompts")
from eval_utils import load_variation, is_action_failed, load_gold_path, get_rm_path, rooms, task_mapping
from prompt_base import (INIT_WEBSHOP,INIT_SCIWORLD,INIT_ALFWORLD,INIT_TEXTCRAFT,INIT_WEBARENA,INIT_BABYAI,
                         SCI_RULE,
                         ACTION_WEBSHOP,ACTION_SCIWORLD,ACTION_TEXTCRAFT,ACTION_ALFWORLD,ACTION_WEBARENA,ACTION_BABYAI,
                         SCIWORLD_TEMPLATE_ADA, SCIWORLD_TEMPLATE_NO_HIS_ADA,
                         SCIWORLD_TEMPLATE_ADA_V2, SCIWORLD_TEMPLATE_NO_HIS_ADA_V2,
                         SCIWORLD_TEMPLATE_NO_HIS, SCIWORLD_TEMPLATE,
                         ALFWORLD_TEMPLATE_ADA, ALFWORLD_TEMPLATE_NO_HIS_ADA,
                         )
from llm_base import vllm, llm_gpt, llm_azure, llm_openai, llm_claude, llm_deepseek, llm_hunyuan

CONTROLLER_ADDR = os.environ.get('CONTROLLER_ADDR', '').split(',')


def clean(s):
    clean_toks = ['\n', '\t']
    for tok in clean_toks:
        s = s.replace(tok, ' ')
    return s

def getValidActionString(available_actions, inventory):
    global rooms
    possible_actions = available_actions.get("possible_actions", [])
    possible_objects = available_actions.get("possible_objects", [])
    valid_open_door = list(set(rooms) & set(possible_objects))
    invalid_rooms = [room for room in rooms if room not in possible_objects]
    possible_actions.append('open OBJ' if valid_open_door else None)
    possible_objects.extend(valid_open_door)
    if "go OBJ" in possible_actions:
        possible_actions.append("go to OBJ")
    inventory = [i.strip() for i in inventory.split("\n\t")[1:] if i]
    
    return ACTION_SCIWORLD.format(
        possible_actions=', '.join(np.unique(possible_actions).tolist()),
        possible_objects=', '.join(np.unique(possible_objects).tolist()),
        inventory=', '.join(inventory)
    ) + '\n' + SCI_RULE.format(invalid_rooms=', '.join(invalid_rooms))


def get_file_name(args):
    if (len(args["output_path"]) > 0):
        args["output_path"] = args["output_path"] + "/"

        # Make path if it doesn't exist
        filenameOutPrefixSeed = f"{args['output_path']}/{str(int(time.time()))}"
        if (not os.path.exists(filenameOutPrefixSeed)):
            try:
                os.makedirs(filenameOutPrefixSeed)
            except:
                pass

    return filenameOutPrefixSeed


def extract_action_from_output(output):
    """从输出中提取action（匹配训练格式）"""
    # 尝试匹配 <action>...</action>
    match = re.search(r'<action>\s*(.+?)\s*</action>', output, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # 兼容旧格式
    markers = [
        ('<|begin_of_answer|>', '<|end_of_answer|>'), 
        ('|begin_of_answer|', '|end_of_answer|')
    ]
    for start_marker, end_marker in markers:
        try:
            if start_marker in output and end_marker in output:
                after_start = output.split(start_marker)[1]
                content = after_start.split(end_marker)[0]
                return content.strip()
        except (IndexError, AttributeError):
            continue
    
    # 最后尝试提取 Action: 后面的内容
    if 'Action:' in output:
        return output.split("Action:")[-1].strip()
    
    return output.strip()


def format_history_for_eval(history_buffer):
    """
    格式化历史记录，匹配训练格式
    history_buffer: List of dicts with keys 'observation', 'action', 'full_output'
    """
    if not history_buffer:
        return ""
    
    action_history = ""
    
    # 显示所有历史步骤：Observation + Action
    for i, step in enumerate(history_buffer):
        step_number = i + 1
        obs = step['observation']
        action = step['action']
        action_history += f"\n[Step {step_number}, Observation {step_number}: '{obs}', Action {step_number}: '{action}']"
    
    # 添加最近3步的reasoning
    if len(history_buffer) > 0:
        recent_length = min(3, len(history_buffer))
        start_idx = len(history_buffer) - recent_length
        action_history += "\n- recent reasoning process: \n"
        
        for i in range(start_idx, len(history_buffer)):
            step_number = i + 1
            full_output = history_buffer[i]['full_output']
            action_history += f"[Step {step_number}, output {step_number}: '{full_output}']\n"
    
    return action_history.strip()


def build_prompt_for_eval(mode, task_description, history_buffer, current_observation, step_count, env):
    """
    构建推理时的 prompt，匹配训练格式
    """
    # 根据环境类型选择对应的模板
    if env == "sciworld":
        if mode == 0:
            _TEMPLATE_NO_HIS = SCIWORLD_TEMPLATE_NO_HIS
            _TEMPLATE = SCIWORLD_TEMPLATE
        else:
            _TEMPLATE_NO_HIS = SCIWORLD_TEMPLATE_NO_HIS_ADA_V2
            _TEMPLATE = SCIWORLD_TEMPLATE_ADA_V2
    elif env == "alfworld":
        _TEMPLATE_NO_HIS = ALFWORLD_TEMPLATE_NO_HIS_ADA
        _TEMPLATE = ALFWORLD_TEMPLATE_ADA

    if not history_buffer:
        # 第一步：使用 NO_HIS 模板
        prompt = _TEMPLATE_NO_HIS.format(
            task_description=task_description,
            current_observation=current_observation
        )
    else:
        # 后续步骤：使用完整模板
        action_history = format_history_for_eval(history_buffer)
        history_length = len(history_buffer)
        
        prompt = _TEMPLATE.format(
            task_description=task_description,
            step_count=step_count,
            history_length=history_length,
            action_history=action_history,
            current_step=step_count + 1,
            current_observation=current_observation
        )
    
    return prompt


# Example user input console, to play through a game.
def eval(args, logger):

    # Initialize environment
    env_args = {
    "env_server_base": args["env_server_base"],
    "data_len": 200,
    "timeout": 300  
    }  
    assert args['env'] in ["sciworld", "alfworld", "webshop", "textcraft", "webarena", "babyai"]
    if args['env'] == "sciworld":
        env = SciworldEnvClient(**env_args)
    elif args['env'] == "alfworld":
        env = AlfWorldEnvClient(**env_args)
    elif args['env'] == "webshop":
        env = WebshopEnvClient(**env_args)
    elif args['env'] == "webarena":
        env = WebarenaEnvClient(**env_args)
    elif args['env'] == "babyai":
        env = BabyAIEnvClient(**env_args)
    else:
        env = TextCraftEnvClient(**env_args)
    
    variations = load_variation(args, logger)
    if os.path.exists(f"{args['output_path']}/final_result.json"):  
        with open(f"{args['output_path']}/final_result.json", 'r') as f:
            prev_results = json.load(f)
        prev_variations = prev_results["variations"]
        variations = [v for v in variations if v not in prev_variations]
    
    filenameOutPrefixSeed = get_file_name(args)

    scores = []
    task_names = []
    _variations = []
    
    for variation in variations:
        env_id = 0
        variation_to_save = f"{filenameOutPrefixSeed}/variation-{variation}"
        try:
            if args['env'] == "sciworld":
                initial_state = env.reset(data_idx=variation) if variation <= 1000 else env.reset(data_idx=variation+300)
                task_name = initial_state['task_name']
                task_description = initial_state['task_description']
                obs = initial_state['observation']
            elif args['env'] == "webshop":
                initial_state = env.reset(idx=variation)
                task_name = ""
                task_description = initial_state[0]
                obs = ""
            elif args['env'] == "alfworld":
                initial_state = env.reset(game=variation)
                task_name = initial_state['task_type']
                task_description = env.observe()
                obs = ""
            elif args['env'] == "textcraft":
                initial_state = env.reset(idx=variation)
                task_name = ""
                task_description = initial_state['observation']
                obs = ""
            elif args['env'] == "babyai":
                initial_state = env.reset(data_idx=variation)            
                task_name = ""
                task_description = initial_state['observation']
                obs = initial_state['observation']
            elif args['env'] == "webarena":
                initial_state = env.reset(idx=variation)
                task_name = ""
                task_description = initial_state[0]
                obs = ""

            if args["gold_path"]:
                gold_path, gold_path_string = load_gold_path(args["env"], variation)
            else:
                gold_path, gold_path_string = [], ""
        except:
            logger.info(f"Variation {variation} failed to load, skip")
            continue

        done = False
        score = 0.0
        last_score = 0.0
        step = 0
        max_steps = args["env_step_limit"] * 2

        # 使用历史缓冲区代替 conversation
        history_buffer = []  # 存储 {observation, action, full_output}
        
        # 构建初始任务描述
        new_task = f'{task_description}\n' + clean(obs) if obs != task_description else f'{task_description}\n'
        current_observation = new_task.strip()

        # Kill agent if it provides more than 10 consecutive invalid actions
        fail_counter = 0
        new_item = {
            "task_name": task_name,
            "task_description": new_task,
            "variation": variation,
            "gold_path": gold_path,
            "gold_path_string": gold_path_string
        }
        
        while not done:
            # 构建 prompt（匹配训练格式）
            prompt_text = build_prompt_for_eval(
                mode=args["thinking_mode"],
                task_description=new_task,
                history_buffer=history_buffer,
                current_observation=current_observation,
                step_count=step,
                env=args["env"]
            )
            
            logger.info("###Prompt###\n" + prompt_text)
            
            # System prompt 为空，所有内容在 user message 中
            prompt = [{"role": "user", "content": prompt_text}]

            # 调用模型
            if 'gpt' in args["actor_model"]:
                action = (llm_openai(prompt, "gpt-4o-nlp") or 
                        vllm(prompt, 'llama3-8b', port=8031) or
                        "<level>1</level>\n<action>look around</action>")
            elif 'claude' in args["actor_model"]:
                action = (llm_hunyuan(prompt, "api_anthropic_claude-3-7-sonnet-20250219") or
                        llm_azure(prompt) or 
                        llm_openai(prompt, "gpt-4o-nlp") or 
                        vllm(prompt, 'llama3-8b', port=8031) or
                        "<level>1</level>\n<action>inventory</action>")
            elif 'deepseek' in args["actor_model"]:
                action = (llm_deepseek(prompt, model=args["actor_model"]) or
                        "<level>1</level>\n<action>inventory</action>")
            elif 'gemini' in args['actor_model']:
                action = llm_hunyuan(prompt, model=args["actor_model"])
            else:
                action = vllm(prompt, model=args["actor_model"], port=args['actor_port'])
            
            logger.info('###Response###\n' + action)

            # 提取 action（从 <action>...</action> 标签中）
            extract_action = extract_action_from_output(action)
            extract_action = extract_action.replace("green house", "greenhouse")
            # 特殊处理 textcraft
            if args["env"] == "textcraft":
                if "from inventory or environment" in extract_action:
                    extract_action = extract_action.replace("from inventory or environment", "").strip()
                if "from inventory" in extract_action or "from the inventory" in extract_action:
                    extract_action = extract_action.replace("from inventory", "").replace("from the inventory", "").strip()
                elif "from environment" in extract_action or "from the environment" in extract_action:
                    extract_action = extract_action.replace("from environment", "").replace("from the environment", "").strip()
            
            # 执行动作
            step_output = env.step(extract_action)
            _, score, done = (
                step_output.state,
                step_output.reward,
                step_output.done,
            )
            obs = env.observe()
            
            # 检查是否失败
            if is_action_failed(obs):
                fail_counter += 1
                if fail_counter >= 10:
                    logger.info('Early stop due to consecutive invalid actions')
                    break
            else:
                fail_counter = 0
            
            if score < 0:
                if args["no_stop"]:
                    done = True
                    score = last_score
                else:
                    done = True
                    score = 0
            
            last_score = score
            obs = clean(obs)
            
            print(action)
            print(obs)
            
            # 保存到历史缓冲区
            history_buffer.append({
                'observation': current_observation,
                'action': extract_action,
                'full_output': action  # 保存完整输出（包含 <level><think><action>）
            })
            
            # 更新当前观察
            current_observation = obs
            
            new_item.update({
                "path_id": step,
                "next_step": action,
                "next_obs": obs,
                "score": score
            })
            
            # 保存轨迹
            with open(f'{variation_to_save}.jsonl', 'a') as f: 
                f.write(json.dumps(new_item) + "\n") 
            
            logger.info(f"Variation: {variation}, Step: {step}, Action: {extract_action}")
            logger.info("Obs: " + obs)
            logger.info(f"Score: {score}")
            logger.info("")

            step += 1
            if (step >= max_steps) or done:
                break

            # Early stopping if we're in a loop
            if len(history_buffer) >= 5:
                recent_actions = [h['action'] for h in history_buffer[-5:]]
                if len(set(recent_actions)) == 2:
                    logger.info("Model is likely in a loop, stopping early.")
                    break
            
            time.sleep(1)

        # Store results
        scores.append(score)
        task_names.append(task_name)
        _variations.append(variation)

        logger.info("Run completed...")
        logger.info("Scores: " + str(scores))
 
        time.sleep(5)
        
        with open(f"{filenameOutPrefixSeed}/final_result.json", 'w') as f: 
            json.dump({"task_names": task_names, "variations": _variations, "scores": scores, "args": args}, f, indent=4)

    # Episodes are finished
    avg = sum(scores) / len(scores) if scores else 0
    logger.info("Average score: " + str(avg))

    f = open(f"{filenameOutPrefixSeed}/score.txt", "a")
    f.write("\n" + "TaskNames:" + str(task_names) + "Variations:" + str(variations) + "Scores: " + str(scores) + " Average score: " + str(avg) + " Args: " + str(args) + "\n")
    f.close()

    logger.info("Shutting down server...")
    logger.info("Completed.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jar_path", type=str, default="") 
    parser.add_argument("--env", default="sciworld")
    parser.add_argument("--env_step_limit", type=int, default=100)
    parser.add_argument("--thinking_mode", type=int, default=0)
    parser.add_argument("--env_server_base", type=str, default="http://11.214.148.199:8401")
    parser.add_argument("--set", default="test")
    parser.add_argument("--output_path", default="")
    parser.add_argument("--no_stop", action="store_true", default=True)
    parser.add_argument("--actor_model", type=str, default="gpt-4")
    parser.add_argument("--actor_port", type=int, default=8401)
    parser.add_argument("--gold_path", action='store_true')

    args = parser.parse_args()
    params = vars(args)
    return params


def init_logger(args, log_level=INFO):
    filenameOutPrefixSeed = get_file_name(args)
    logger = logging.getLogger()
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s\t] %(message)s",
                                    datefmt='%Y-%m-%d %H:%M:%S')
    logger.setLevel(log_level)

    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logging_dir = args["output_path"]
    if logging_dir:
        os.makedirs(logging_dir, exist_ok=True)
        filename = f"{filenameOutPrefixSeed}.log"
        fh = logging.FileHandler(filename)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        if logger.hasHandlers():
            logger.handlers.clear()
        logger.addHandler(fh)
    return logger

def main():
    args = parse_args()
    print(args)  
    logger = init_logger(args)
    logger.info(args)
    eval(args, logger)
        
if __name__ == "__main__":
    main()