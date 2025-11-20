import argparse
import json
import logging
import os
import ast
import time
import pdb
import sys
import requests
import tiktoken
import random
import numpy as np
from logging import INFO
from typing import Dict, List
from openai import OpenAI
from fastchat.conversation import Conversation, SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template
from requests.exceptions import Timeout
from agentenv.envs import SciworldEnvClient, AlfWorldEnvClient, WebshopEnvClient, TextCraftEnvClient, BabyAIEnvClient
sys.path.append("/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/utils")
from eval_utils import load_variation, get_trajectory
from prompt_base import (
            INSTRUCT_CRITIC, 
            INSTRUCT_GOLD_ACTION_CRITIC, 
            SCI_RULE,
            INIT_WEBSHOP, 
            INIT_SCIWORLD, 
            INIT_ALFWORLD,
            INIT_TEXTCRAFT,
            INIT_BABYAI,
            ACTION_SCIWORLD,
            ACTION_WEBSHOP,
            ACTION_TEXTCRAFT,
            ACTION_BABYAI,
            ACTION_ALFWORLD,
            INSTRUCT_PARSE,
            # Import thinking modes from second code
            THINK_MODE_0,
            THINK_MODE_1,
            THINK_MODE_2,
            THINK_MODE_3,
            THINK_MODE_4,
            THINK_MODE_5,
            INSTRUCT_GENERATE_ADAPTIVE_THINKING,
            INSTRUCT_GENERATE_LEVEL_4_THINKING,
            )
from llm_base import vllm, llm_azure, llm_hunyuan

INIT_PROMPT = {"sciworld": INIT_SCIWORLD, "webshop": INIT_WEBSHOP, "alfworld": INIT_ALFWORLD, "textcraft": INIT_TEXTCRAFT, "babyai": INIT_BABYAI}

# Add thinking modes dictionary
THINK_MODES = {
    0: THINK_MODE_0,  # base
    1: THINK_MODE_1,
    2: THINK_MODE_2, 
    3: THINK_MODE_3,
    4: THINK_MODE_4,
    5: THINK_MODE_5
}

# Add prompt template for generating thinking process with adaptive level selection


CONTROLLER_ADDR = os.environ.get('CONTROLLER_ADDR', '').split(',')


def clean(s):
    clean_toks = ['\n', '\t']
    for tok in clean_toks:
        s = s.replace(tok, ' ')
    return s

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

    
def get_prompt(conv: Conversation) -> str:
    if conv.name == 'openchat':
        ret = ''
        for role, message in conv.messages:
            if message:
                # ret += role + ": " + message + conv.sep
                print(role, message)
                ret += role + ": " + message
            else:
                ret += role + ":"  
        return ret
    else:
        return conv.get_prompt()

def generate_adaptive_thinking_process(mode, task_description, history, current_obs, gold_action):
    """Generate thinking process with adaptive level selection based on gold action"""
    
    # Extract just the action part if gold_action contains "Action:"
    clean_gold_action = gold_action.split("Action:")[-1].strip() if "Action:" in gold_action else gold_action.strip()
    
    if mode == 5:
        thinking_instruction = INSTRUCT_GENERATE_ADAPTIVE_THINKING.format(
            task_description=task_description,
            history=history,
            current_obs=current_obs,
            gold_action=clean_gold_action
        )
    elif mode == 4:
        thinking_instruction = INSTRUCT_GENERATE_LEVEL_4_THINKING.format(
            task_description=task_description,
            history=history,
            current_obs=current_obs,
            gold_action=clean_gold_action
        )
    
    thinking_prompt = [{"role": "user", "content": thinking_instruction}]
    thinking_response = llm_azure(thinking_prompt)
    # thinking_response = llm_hunyuan(prompt=thinking_prompt, model="api_azure_openai_gpt-4o")
    
    # Fallback response if LLM fails
    if not thinking_response:
        return f"Thinking Level: 1\n<|begin_of_answer|>\nAction: {clean_gold_action}\n<|end_of_answer|>"
    
    return thinking_response
    

# Example user input console, to play through a game.
def eval(args, logger):

    # Initialize environment
    env_args = {
    "env_server_base": args["env_server_base"],
    "data_len": 200,
    "timeout": 300 
    }  
    assert args['env'] in ["sciworld", "alfworld", "webshop", "textcraft", "babyai"]
    if args['env'] == "sciworld":
        env = SciworldEnvClient(**env_args)
    elif args['env'] == "alfworld":
        env = AlfWorldEnvClient(**env_args)
    elif args['env'] == 'webshop':
        env = WebshopEnvClient(**env_args)
    elif args['env'] == 'textcraft':
        env = TextCraftEnvClient(**env_args)
    elif args['env'] == "babyai":
        env = BabyAIEnvClient(**env_args)

    variations = load_variation(args, logger)
    if args["debug"]:
        random.seed(42)
        variations = random.sample(variations, 20)
    
    if os.path.exists(f"{args['output_path']}/final_result.json"):  
        with open(f"{args['output_path']}/final_result.json", 'r') as f:
            prev_results = json.load(f)
        prev_variations = prev_results["variations"]
        variations = [v for v in variations if v not in prev_variations]
        
    filenameOutPrefixSeed = get_file_name(args)

    # Load encoding tool to count token numbers
    token_model = 'gpt-4'
    encoding = tiktoken.encoding_for_model(token_model)

    scores = []
    task_names = []
    _variations = []
    
    for variation in variations:
        # try:
            env_id = 0
            for repeat in range(args['n_repeat']):
                dir_to_save = f"{filenameOutPrefixSeed}/variation-{variation}"
                if (not os.path.exists(dir_to_save)):
                    os.makedirs(dir_to_save)
                variation_to_save = f"{dir_to_save}/repeat-{repeat}"

                # train_data = []
                if args['env'] == "sciworld":
                    initial_state = env.reset(data_idx=variation) if variation <= 1000 else env.reset(data_idx=variation+300)
                    task_name = initial_state['task_name']
                    task_description = initial_state['task_description']
                    recent_actions = ["look around"]
                    obs = initial_state['observation']
                elif args['env'] == "webshop":
                    initial_state = env.reset(idx=variation)
                    task_name = ""
                    task_description = initial_state[0]
                    recent_actions = []
                    obs = ""
                elif args['env'] == "alfworld":
                    initial_state = env.reset(game=variation)
                    task_name = initial_state['task_type']
                    task_description = env.observe()
                    recent_actions = []
                    obs = ""
                elif args['env'] == "textcraft":
                    initial_state = env.reset(idx=variation)
                    task_name = ""
                    task_description = initial_state['observation']
                    recent_actions = []
                    obs = ""
                elif args['env'] == "babyai":
                    initial_state = env.reset(data_idx=variation)
                    task_name = ""
                    task_description = initial_state['observation']
                    recent_actions = []
                    obs = ""
                    task_example = ""
                
                with open(f"/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/data/{args['env']}_goldpath.json") as f:
                    gold_path_all = json.load(f)
                gold_actions = [step['action'] for step in gold_path_all[f"{variation}"]["path"]]
                observations = [step['observation'] for step in gold_path_all[f"{variation}"]["path"]]
                task_description = gold_path_all[f"{variation}"]["taskDescription"]
                
                done = False
                score = 0.0
                last_score = 0.0
                step = 0

                # The env has an internal step count, some actions like look around are free
                # however, the t5 model only generates the action "look around", which will result in a dead loop below
                # so the max_steps here is only used to avoid the model generating the same action forever
                max_steps = args["env_step_limit"] * 2

                conv = get_conversation_template('gpt-4o')
                conv.set_system_message("You are a helpful, respectful and honest assistant.")
                
                # Include thinking mode in initial prompt (use mode 5 for adaptive thinking)
                init_prompt = INIT_PROMPT[args['env']] + '\n' + THINK_MODE_5
                
                conv.append_message(conv.roles[0], init_prompt)
                if args['env'] in ["sciworld", "alfworld", "textcraft"]:
                    conv.append_message(conv.roles[1], "OK. I'll follow your instructions and try my best to solve the task.")
                else:
                    conv.append_message(conv.roles[1], "Ok.")

                new_task = f'{task_description}\n' + clean(obs) 
                conv.append_message(conv.roles[0], new_task.strip())

                max_len = 4096
                new_item = {
                    "task_name": task_name,
                    "task_description": new_task,
                    "variation": variation
                    }
                
                for gold_action, obs in zip(gold_actions, observations):
                    while len(encoding.encode(get_prompt(conv))) > max_len - 60:
                        # Remove the oldest actions in the few-shot
                        del conv.messages[5:7]
                        
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.to_openai_api_messages()
                    logger.info("###Prompt###\n" + get_prompt(conv))
                    pdb.set_trace()
                    # Add the observation at the beginning for current state context
                    if step == 0:
                        # obs = initial_state.get('observation', '')
                        current_obs = new_task.strip()
                    else:
                        current_obs = obs.strip()
                
                    # Generate adaptive thinking process for the gold action
                    command, history = get_trajectory(prompt, task_description)
                    # current_obs = obs  # Current observation for context
                    thinking_process = generate_adaptive_thinking_process(
                        mode=args['thinking_mode'],
                        task_description=task_description,
                        history=history,
                        current_obs=current_obs,
                        gold_action=gold_action
                    )
                    
                    
                    # Use the generated thinking process instead of just the gold action
                    logger.info('###Response with Thinking###\n' + thinking_process)
                    
                    conv.update_last_message(thinking_process)
                    
                    # Extract the action part for execution
                    clean_gold_action = gold_action.split("Action:")[-1].strip() if "Action:" in gold_action else gold_action.strip()
                    step_output = env.step(clean_gold_action)
                    _, score, done = (
                        step_output.state,
                        step_output.reward,
                        step_output.done,
                    )
                    obs = env.observe()
                    last_score = score
                    
                    obs = clean(obs)
                    print(obs)
                    
                    new_item.update({
                        "path_id": step,
                        "prompt": prompt,
                        "gold_action": gold_action,
                        "thinking_process": thinking_process,
                        "next_step": thinking_process,  # Save full thinking process
                        "next_obs": obs,
                        "score": score
                    })
                    
                    # Add action and observation to game prompt
                    conv.append_message(conv.roles[0], obs)
                    recent_actions.append(f'({clean_gold_action}, {obs})')
                    
                    with open(f'{variation_to_save}.jsonl', 'a') as f: 
                        f.write(json.dumps(new_item) + "\n") 
                    
                    #logger.info("Input string: " + str(input_str))
                    logger.info(f"Variation: {variation}, Step: {step}, Action: {clean_gold_action}")
                    logger.info("Obs: " + obs)
                    logger.info(f"Score: {score}")
                    logger.info("")

                    step += 1
                    if (step >= max_steps) or done:
                        break
        
                    logger.info("Recent Actions: " + str(recent_actions))

                    # Early stopping if we're in a loop
                    if len(recent_actions) >= 5 and len(set(recent_actions[-5:])) == 2:
                        logger.info("Many recent actions in history are the same -- model is likely in a loop, stopping early.")
                        break
                    
                    time.sleep(2)

                # Store results
                scores.append(score)
                task_names.append(task_name)
                _variations.append(variation)

                logger.info("Run completed...")
                logger.info("Scores: " + str(scores))
        
                time.sleep(5)
            
                with open(f"{filenameOutPrefixSeed}/final_result.json", "w") as f:
                    json.dump({"task_names": task_names, "variations": _variations, "scores": scores, "args": args}, f, indent=4)
                
                if score != 1:
                    break
        # except Exception as e:
            # logger.error(f"Error processing variation {variation}: {e}, continuing to next variation...")
            # continue
            

    # Episodes are finished -- manually save any last histories still in the buffer
    avg = sum(scores) / len(scores)
    logger.info("Average score: " + str(avg))

    f = open(f"{filenameOutPrefixSeed}/score.txt", "a")
    f.write("\n" + "TaskNames:" + str(task_names) + "Variations:" + str(variations) + "Scores: " + str(scores) + " Average score: " + str(avg) + " Args: " + str(args) + "\n")
    f.close()
    logger.info("Shutting down server...")
    # env.shutdown()

    logger.info("Completed.")



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jar_path", type=str, default="") 
    parser.add_argument("--env", default="sciworld")  # use comma to split 
    parser.add_argument("--env_step_limit", type=int, default=100)
    parser.add_argument("--env_server_base", type=str, default="http://11.214.148.199:8401")
    parser.add_argument("--n_repeat", type=int, default=10)
    parser.add_argument("--set", default="test")
    parser.add_argument("--output_path", default="")
    parser.add_argument("--no_stop", action="store_true", default=True)
    parser.add_argument("--actor_model", type=str, default="gpt-4")
    parser.add_argument("--actor_port", type=int, default=8401)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--thinking_mode", type=int, default=5, help="Thinking mode level (0-5, 5=adaptive)")

    args = parser.parse_args()
    params = vars(args)
    return params

#
#   Main
#

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