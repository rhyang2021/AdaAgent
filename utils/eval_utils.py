
# from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoConfig
from math import ceil
import random
import re
import pdb
import json
import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time 
import tiktoken 
import requests
from typing import Dict, List

action_type_description = [
    {"action_type": "WAIT()", "desc": "wait for something to be done, for example, an object on stove to be boiled"},
    {"action_type": "TELEPORT(room)", "desc": "directly go to a room such as TELEPORT(kitchen)"},
    {"action_type": "LOOK(object)", "desc": "look at an object"},
    {"action_type": "READ(object)", "desc": "read an object such as a recipe or a book"},
    {"action_type": "PICK(object)", "desc": "pick up an object and put it into your inventory"},
    {"action_type": "OPEN(object)", "desc": "open an object with doors before you search or put things in it. For example, OPEN(freezer), OPEN(blast furnace)."},
    {"action_type": "ACTIVATE(object)", "desc": "activate and turn on an object such as sink or stove, so that you can use it. "},
    {"action_type": "DEACTIVATE(object)", "desc": "deactivate turn off the object"},
    {"action_type": "EXAMINE(object)", "desc": "look at an object carefully. For example, EXAMINE(apple). Note that you cannot EXAMINE a location."},
    {"action_type": "CONNECT(object)", "desc": "connect two objects so that they become useful"},
    {"action_type": "MOVE(object, place)", "desc": "move/place the object to a place"},
    {"action_type": "USE(object A, object B)", "desc": "use an object A on object B, for example, USE(thermometer in inventory, water) to check the temperature of water."},
    {"action_type": "MIX(container)", "desc": "mix the objects in a container such as MIX(cup containing sugar and water)"},
    {"action_type": "DUNK(object A, object B)", "desc": "dunk object A into object B (optional)"},
    {"action_type": "DROP(object A, object B)", "desc": "drop object A into object B (optional)"},
    {"action_type": "POUR(object A, object B)", "desc": "pour the object A into the container B; For example, POUR(red paint, glass cup)"},
    {"action_type": "FOCUS(object)", "desc": "focus on an important object that are required by the task description (e.g., a substance, a plant, an animal, and so on)."},
]

focus_on_count = {
    "0": 1, "1": 1, "2": 1, "3": 1, "4": 2, "5": 1, "6":1, "7":1,
    "8": 1, "9": 1, "10": 1, "11": 1, "12": 4, "13": 4, "14":1, "15":1,
    "16": 1, "17": 1, "18": 2, "19": 1, "20": 3, "21": 3, "22":1, "23":1,   
    "24": 1, "25": 1, "26": 2, "27": 1, "28": 1, "29": 2
    
}

task_mapping = {
    "boil": "0",
    "melt": "22",
    "freeze": "9",
    "change-the-state-of-matter-of": "1",
    "use-thermometer": "29",
    "measure-melting-point-known-substance": "20",
    "measure-melting-point-unknown-substance": "21",
    "power-component": "25",
    "power-component-renewable-vs-nonrenewable-energy": "26",
    "test-conductivity": "27",
    "test-conductivity-of-unknown-substances": "28",
    "find-living-thing": "6",
    "find-non-living-thing": "7",
    "find-plant": "8",
    "find-animal": "5",
    "grow-plant": "11",
    "grow-fruit": "10",
    "chemistry-mix": "2",
    "chemistry-mix-paint-secondary-color": "3",
    "chemistry-mix-paint-tertiary-color": "4",
    "lifespan-longest-lived": "17",
    "lifespan-shortest-lived": "19",
    "lifespan-longest-lived-then-shortest-lived": "18",
    "identify-life-stages-1": "12",
    "identify-life-stages-2": "13",
    "inclined-plane-determine-angle": "14",
    "inclined-plane-friction-named-surfaces": "15",
    "inclined-plane-friction-unnamed-surfaces": "16",
    "mendelian-genetics-known-plant": "23",
    "mendelian-genetics-unknown-plant": "24"

}

rooms = ["hallway", "greenhouse", "green house", "kitchen", "bathroom", "outside", "workshop", "art studio", "foundry", "bedroom", "living room"]


def load_variation(args, logger):
    with open("/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/data/task_dev.json") as f:
        dev = json.load(f)
    train_set = dev[0]['train']
    train_mini_set = dev[0]['train_mini']
    train_add_set = dev[0]['train_add']
    test_set = dev[0]['test']

    variations = []
    if (args["set"] == "train"):
        variations = train_set[args["env"]]['idxs']
    elif (args["set"] == "train_mini"):
        variations = train_mini_set[args["env"]]['idxs']
    elif (args["set"] == "train_add"):
        variations = train_add_set[args["env"]]['idxs']
    elif (args["set"] == "train_mini2"):
        variations = train_mini_set[args["env"]]['idxs']
        random.seed(1)
        variations = random.sample(variations, min(50, len(variations)))
    elif (args["set"] == "test"):
        variations = test_set[args["env"]]
    elif (args["set"] == "test_mini"):
        variations = test_set[args["env"]]
        random.seed(1)
        variations = random.sample(variations, min(100, len(variations)))
    else:
        logger.info("ERROR: Unknown set to evaluate on (" + str(args["set"]) + ")")
        exit(1)
 
    logger.info(variations)
    return variations

def is_action_failed(obs):
    return "No known action matches that input." == obs or "can't" in obs or "not" in obs or "doesn't" in obs or "Nothing happened" in obs


def load_gold_path(env, variation):
    
    with open(f"/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/data/{env}_goldpath.json") as f:
        gold_path_all = json.load(f)
    gold_path = gold_path_all[f"{variation}"]["path"]
    gold_path_string = ""
    for step in gold_path:
        _action = step['action'].split("Action:")
        if len(_action) > 1:
            action = _action[1].strip()
        else:
            action = _action[0].strip()
        gold_path_string += f"Action: {action}\nObservation: {step['observation']}\n"
    return gold_path, gold_path_string

def pop_k_elements(lst, piece=5):
    k = round(len(lst)/piece)
    result = []
    for _ in range(piece):
        result.append(lst)
        if len(lst) < k:
            break
        lst = lst[:-k]
    return result

def get_trajectory(prompt, task_description):

    cnt = 0
    for traj in prompt:
        if f"{task_description}" in traj['content'] and "The preceding task has been completed" not in traj['content']:
            break
        cnt += 1
    new_task = prompt[cnt]['content']
    if "AVAILABLE ACTIONS:" in new_task:
        new_task = new_task.split("AVAILABLE ACTIONS:")[0].strip()
    history = new_task.replace("Your task", "The agent's task goal") + '\n'

    cur_traj = prompt[(cnt+1): ]
    for i in range(1, len(cur_traj), 2):
        user_msg = cur_traj[i]
        assitant_msg = cur_traj[i-1]
        _action = assitant_msg['content'].split("Action:")
        if len(_action) > 1:
            action = _action[1].split('\n<|end_of_answer|>')[0].strip()
        else:
            action = _action[0].split('\n<|end_of_answer|>')[0].strip()
        obs = user_msg['content']
        if "AVAILABLE ACTIONS:" in obs:
            obs = obs.split("AVAILABLE ACTIONS:")[0].strip()
        history += f"Action: {action}\nObservation: {obs}\n"
    
    if len(history.split("Goal: ")) > 1:
        _history = 'Goal: ' + history.split("Goal: ")[1]
        command = history.split("Goal: ")[0].strip()
    else:
        _history = history
        command = ""

    return command, _history


def get_rm_path(prompt, task_description):
    cnt = 0
    for traj in prompt:
        if f"{task_description}" in traj['content'] and "The preceding task has been completed" not in traj['content']:
            break
        cnt += 1
    cur_traj = prompt[(cnt+1): ]
    history_actions = []
    for i in range(1, len(cur_traj), 2):
        assitant_msg = cur_traj[i-1]
        _action = assitant_msg['content'].split("Action:")
        if len(_action) > 1:
            action = _action[1].strip()
        else:
            action = _action[0].strip()
        history_actions.append(action)
    _actions = [f" {10 - i}. {action}." for i, action in enumerate(history_actions[-10:])]
    return task_description + ' '.join(_actions) + " Action: "


def getValidACtionObjectCombinations(action_hint):
    possible_actions, possible_objects = action_hint['possible_actions'], action_hint['possible_objects']
    validActions = set()

    for action in possible_actions:
        if 'OBJ' in action:
            if action.count('OBJ') == 1:
                for obj in possible_objects:
                    validActions.add(action.replace('OBJ', obj))
                    if action == "go OBJ":
                        validActions.add("go to " + obj)
                
            elif action.count('OBJ') == 2:
                for obj1 in possible_objects:
                    for obj2 in possible_objects:
                        validActions.add(action.replace('OBJ', obj1, 1).replace('OBJ', obj2, 1))
        else:
            validActions.add(action)
    return validActions

def findValidActionNew(predictions, env, action_hint, recent_actions, sbert_model, logger, env_id, k=5):
    global rooms
    valid_open_door = ["open door to " + i for i in rooms if i in action_hint['possible_objects']] 
    
    # invalid_focus = ["focus on "+x for x in ["agent", "air"]+rooms]
    server_url = env.env_server_base
    validActions = getValidACtionObjectCombinations(action_hint)
    validActions.update(valid_open_door)
    # validActions.difference_update(invalid_focus)
    inventory =  env.step("inventory").state.lower()
    # validActions.difference_update(recent_actions[-3:]) 
    # validActions.update("look around")
    response = requests.get(f"{server_url}/observation", params={"id": env_id})
    look = response.json()

    
    for va in list(validActions):
        if "door" in va and "open" not in va:
            validActions.remove(va)
            continue
       
        if va.startswith("focus on"): 
            pattern = re.compile(r"\b(?:focus|on|in|to)\b", re.IGNORECASE)
            used_objs = pattern.sub("", va).split(" ")
            valid = True
            for obj in used_objs:
                if obj not in look + " " + inventory:
                    valid = False
            if not valid:
                validActions.remove(va)
    

    # 1) if acton in top k is valid, choose it
    found_valid_in_top = False
    action = None
    for pred in predictions[:k]:
        pred = pred.replace("green house", "greenhouse") 
        if pred.strip() in validActions:
            found_valid_in_top = True
            action = pred.strip()
            break
    
    action = predictions[0]
    if found_valid_in_top:
        return action
    
    elif "go to" in action:
        if "open door to" + action.replace("go to", "").strip() in validActions:
            env.step("open door to" + action.replace("go to", ""))
            return action
        elif "open door to hallway" in validActions:
            env.step("open door to hallway")
            return "go to hallway" 
    else:
        logger.info(f"No valid action found in top k={k} predictions.")
        validActions = list(validActions)
        validActions.sort(key=lambda x: len(x))
        logger.info("Valid Predictions: "+ str(validActions)) 

    # 2) else, find most similar action

    if sbert_model:    
        
        
        pred_vectors = sbert_model.encode(predictions[:5], batch_size=5, show_progress_bar=False)
        valid_action_vectors = sbert_model.encode(validActions, batch_size=min(len(validActions), 128), show_progress_bar=False)


        # Calculate cosine similarity between each vector in pred_vectors and all vectors in valid_action_vectors
        similarity_matrix = cosine_similarity(pred_vectors, valid_action_vectors)

        # Take the sum of cosine similarities for each vector in valid_action_vectors
        sum_similarities = similarity_matrix.sum(axis=0)

        # Find the indices of the k vectors with the highest sum of cosine similarities
        N = 5 # Change this to the number of top vectors you want to retrieve
        top_indices = np.argpartition(sum_similarities, -N)[-N:]

        # Print the indices of the top vectors
        # print(f"The indices of the top {k} vectors in valid_action_vectors are: {top_indices}")
        logger.info("The most similar valid actions to the predictions:")
        for ti in top_indices:
            logger.info("\t\t - "+validActions[ti])
        action = validActions[top_indices[0]]
    else:
        # jaccard
        topValue = 0.0
        topAction = predictions[0]
        # embPred = sbert_model.encode(pred, convert_to_tensor=True)
        tokensPred = predictions[0].split(" ")
        uniqueTokensPred = set(tokensPred)

        for validAction in validActions: 
            tokensAction = validAction.split(" ")
            uniqueTokensAction = set(tokensAction)

            intersection = uniqueTokensPred.intersection(uniqueTokensAction)
            if (len(intersection) > topValue):
                topAction = validAction
                topValue = len(intersection)

        logger.info("TOP VALID ACTION: " + topAction)
        # Sanitize top action
        topAction = re.sub(r'[^A-Za-z0-9 ]+', '', topAction)
        action = topAction
    return action
 