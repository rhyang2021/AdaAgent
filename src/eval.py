 
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
from fastchat.conversation import Conversation, SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template
from requests.exceptions import Timeout
from agentenv.envs import SciworldEnvClient, AlfWorldEnvClient, WebshopEnvClient, TextCraftEnvClient, WebarenaEnvClient, BabyAIEnvClient
sys.path.append("/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/utils")
sys.path.append("/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/prompts")
from eval_utils import load_variation, is_action_failed, load_gold_path, get_rm_path, rooms, task_mapping
from prompt_base import (INSTRUCT_CRITIC, 
                         INSTRUCT_GOLD_PATH_CRITIC, 
                         INIT_WEBSHOP,INIT_SCIWORLD,INIT_ALFWORLD,INIT_TEXTCRAFT,INIT_WEBARENA,INIT_BABYAI,
                         THINK_MODE_1,THINK_MODE_2,THINK_MODE_3,THINK_MODE_4, THINK_MODE_0, THINK_MODE_5,THINK_MODE_BASE,
                         SCI_RULE,
                         ACTION_WEBSHOP,ACTION_SCIWORLD,ACTION_TEXTCRAFT,ACTION_ALFWORLD,ACTION_WEBARENA,ACTION_BABYAI,
                         )
from llm_base import vllm, llm_gpt, llm_azure, llm_openai, llm_claude, llm_deepseek, llm_hunyuan

INIT_PROMPT = {"sciworld": INIT_SCIWORLD,
                "webshop": INIT_WEBSHOP,
                "webarena": INIT_WEBARENA,
                "alfworld": INIT_ALFWORLD, 
                "textcraft": INIT_TEXTCRAFT, 
                "babyai": INIT_BABYAI
               }

THINK_MODES = {
    0: THINK_MODE_0,  # base
    1: THINK_MODE_1,
    2: THINK_MODE_2, 
    3: THINK_MODE_3,
    4: THINK_MODE_4,
    5: THINK_MODE_5,
    6: THINK_MODE_BASE
}

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

def get_demostration(examples, task_id):
    prompts = examples[task_id]
    lines = ["Task description:", prompts[0]]
    for prompt in prompts[1:]:
        if 'Think:' in prompt:
            lines.append("Thought:")
            lines.append(prompt.replace('Think: ', '').strip())
        elif 'Action:' in prompt:
            lines.append("Action:")
            lines.append(prompt.replace('Action: ', '').strip())
        elif 'OK.' in prompt:
            continue
        else:
            lines.append(f"Observation: {prompt}")
    return '\n'.join(lines) + '\n'

    
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


def extract_response(text):
    markers = [
        ('<|begin_of_answer|>', '<|end_of_answer|>'), 
        ('|begin_of_answer|', '|end_of_answer|'),
        ('<action>', '</action>')     
    ]
    for start_marker, end_marker in markers:
        try:
            if start_marker in text and end_marker in text:
                after_start = text.split(start_marker)[1]
                content = after_start.split(end_marker)[0]
                return content.strip()
        except (IndexError, AttributeError):
            continue
    return text


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

    # Load encoding tool to count token numbers
    token_model = 'gpt-4'
    encoding = tiktoken.encoding_for_model(token_model)

    scores = []
    task_names = []
    _variations = []
    
    for variation in variations:
        env_id = 2
        variation_to_save = f"{filenameOutPrefixSeed}/variation-{variation}"
        try:
            if args['env'] == "sciworld":
                initial_state = env.reset(data_idx=variation) if variation <= 1000 else env.reset(data_idx=variation+300)
                task_name = initial_state['task_name']
                task_description = initial_state['task_description']
                recent_actions = ["look around"]
                obs = initial_state['observation']
                
                # with open("/apdcephfs_cq10/share_1567347/share_info/ruihanyang/GenRM/prompts/prompt.json") as f:
                    # examples = json.load(f)
                # task_example = get_demostration(examples, task_mapping[task_name])
                task_example = ""
            elif args['env'] == "webshop":
                initial_state = env.reset(idx=variation)
                task_name = ""
                task_description = initial_state[0]
                recent_actions = []
                obs = ""
                task_example = ""
            elif args['env'] == "alfworld":
                initial_state = env.reset(game=variation)
                task_name = initial_state['task_type']
                task_description = env.observe()
                recent_actions = []
                obs = ""
                task_example = ""
            elif args['env'] == "textcraft":
                initial_state = env.reset(idx=variation)
                task_name = ""
                task_description = initial_state['observation']
                recent_actions = []
                obs = ""
                task_example = ""
            elif args['env'] == "babyai":
                initial_state = env.reset(data_idx=variation)            
                task_name = ""
                task_description = initial_state['observation']
                recent_actions = []
                obs = initial_state['observation']
                task_example = ""
            elif args['env'] == "webarena":
                initial_state = env.reset(idx=variation)
                task_name = ""
                task_description = initial_state[0]
                recent_actions = []
                obs = ""
                task_example = ""

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
        
        # The env has an internal step count, some actions like look around are free
        # however, the t5 model only generates the action "look around", which will result in a dead loop below
        # so the max_steps here is only used to avoid the model generating the same action forever
        max_steps = args["env_step_limit"] * 2

        conv = get_conversation_template('gpt-4o')
        conv.set_system_message("You are a helpful, respectful and honest assistant.")
        
        init_prompt = INIT_PROMPT[args['env']] + '\n' + THINK_MODES[args['thinking_mode']]

        conv.append_message(conv.roles[0], init_prompt)
        if args['env'] in ["sciworld", "alfworld", "textcraft", "babyai"]:
            conv.append_message(conv.roles[1], "OK. I'll follow your instructions and try my best to solve the task.")
        else:
            conv.append_message(conv.roles[1], "Ok.")

        new_task = f'{task_description}\n' + clean(obs) if obs !=task_description else f'{task_description}\n'

        conv.append_message(conv.roles[0], new_task.strip())

        max_len = 4096
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
            while len(encoding.encode(get_prompt(conv))) > max_len - 60:
                # Remove the oldest actions in the few-shot
                del conv.messages[5:7]
                
            conv.append_message(conv.roles[1], None)
            prompt = conv.to_openai_api_messages()
            logger.info("###Prompt###\n" + get_prompt(conv))
            if args['env'] == "sciworld":
                response = requests.get(f"{env.env_server_base}/action_hint", params={"id": env_id})
                available_actions = response.json()
                inventory =  env.step("inventory").state
                available_actions_string = getValidActionString(available_actions, inventory)
            elif args['env'] == "alfworld":
                valid_actions = '\n'+'\n'.join([f"- **{action}**" for action in env.info['available_actions']])+'\n'
                available_actions_string = ACTION_ALFWORLD.format(valid_actions=valid_actions)
            elif args['env'] == "babyai":
                valid_action_list = ast.literal_eval(env.observe().split("Available actions: ")[1].strip())
                valid_actions = '\n'+'\n'.join(valid_action_list)+'\n'
                available_actions_string = ACTION_BABYAI.format(valid_actions=valid_actions) 
            elif args['env'] == "webshop":
                available_actions_string = ACTION_WEBSHOP
            elif args['env'] == "webarena":
                available_actions_string = ACTION_WEBARENA
            else:
                available_actions_string = ACTION_TEXTCRAFT 

            if 'gpt' in args["actor_model"]:
                action = (llm_hunyuan(prompt) or
                        "Action: inventory")
            elif 'openai-o3' in args["actor_model"]:
                action = (llm_hunyuan(prompt) or
                        "Action: inventory")
            elif 'claude' in args["actor_model"]:
                action = (llm_hunyuan(prompt) or
                        llm_azure(prompt) or 
                        llm_openai(prompt, "gpt-4o-nlp") or 
                        llm_openai(prompt, "gpt-4-1106-preview-nlp") or 
                        vllm(prompt, 'llama3-8b', port=8031) or
                        "Action: inventory")
            elif "gemini" in args["actor_model"]:
                action = (llm_hunyuan(prompt) or
                        vllm(prompt, 'llama3-8b', port=8031) or
                        "Action: inventory")
            elif 'deepseek' in args["actor_model"]:
                action = (# llm_hunyuan(prompt, "api_doubao_DeepSeek-V3-241226") or
                        llm_deepseek(prompt, model=args["actor_model"]) or
                        "Action: inventory")
            else:
                action = vllm(prompt, model=args["actor_model"], port=args['actor_port'])
                
            
            logger.info('###Response###\n' + action)
            conv.update_last_message(action)

            # extract action from the response
            action = action.replace("**Action:**", "Action:").strip()
            extract_action = extract_response(action)
            if args["env"] == "textcraft":
                if "from inventory or environment" in extract_action:
                    extract_action = extract_action.replace("from inventory or environment", "").strip()
                if "from inventory" in extract_action or "from the inventory" in extract_action:
                    extract_action = extract_action.replace("from inventory", "").replace("from the inventory", "").strip()
                elif "from environment" in extract_action or "from the environment" in extract_action:
                    extract_action = extract_action.replace("from environment", "").replace("from the environment", "").strip()       
            
            # if 'Thought:' in extract_action:
                # extract_action = extract_action.split('Thought:')[0].strip()
            step_output = env.step(extract_action)
            _, score, done = (
                step_output.state,
                step_output.reward,
                step_output.done,
            )
            obs = env.observe()
            # if is_action_failed(obs):
                # fail_counter += 1
                # if fail_counter >= 10:
                    # logger.info('Early stop due to consecutive invalid actions')
                    # break
            # else:
                # fail_counter = 0
            
            if score < 0:
                # Our own solution for dealing with such cases
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
            
            new_item.update({
                "path_id": step,
                "next_step": action,
                "next_obs": obs,
                "score": score
            })
            
            # Add action and observation to game prompt
            conv.append_message(conv.roles[0], obs)
            recent_actions.append(f'({action}, {obs})')
            
            with open(f'{variation_to_save}.jsonl', 'a') as f: 
                f.write(json.dumps(new_item) + "\n") 
            
            #logger.info("Input string: " + str(input_str))
            logger.info(f"Variation: {variation}, Step: {step}, Action: {action}")
            logger.info("Obs: " + obs)
            logger.info(f"Score: {score}")
            logger.info("")

            step += 1
            if (step >= max_steps) or done:
                break
  
            logger.info("Recent Actions: " + str(recent_actions))

            # Early stopping if we're in a loop
            # if len(recent_actions) >= 5 and len(set(recent_actions[-5:])) == 2:
                # logger.info("Many recent actions in history are the same -- model is likely in a loop, stopping early.")
                # break
            
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
    