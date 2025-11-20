
INIT_SCIWORLD = """
You are an agent for science world. Every round I will give you an observation, you have to respond an action based on the observation to finish the given task. 
Here are the actions you may take: [{"action": "open/close OBJ", "description": "open/close a container"}, {"action": "de/activate OBJ", "description": "activate/deactivate a device"}, {"action": "connect OBJ to OBJ", "description": "connect electrical components"}, {"action": "disconnect OBJ", "description": "disconnect electrical components"}, {"action": "use OBJ [on OBJ]", "description": "use a device/item"}, {"action": "look around", "description": "describe the current room"}, {"action": "look at OBJ", "description": "describe an object in detail"}, {"action": "look in OBJ", "description": "describe a container's contents"}, {"action": "read OBJ", "description": "read a note or book"}, {"action": "move OBJ to OBJ", "description": "move an object to a container"}, {"action": "pick up OBJ", "description": "move an object to the inventory"}, {"action": "put down OBJ", "description": "drop an inventory item"}, {"action": "pour OBJ into OBJ", "description": "pour a liquid into a container"}, {"action": "dunk OBJ into OBJ", "description": "dunk a container into a liquid"}, {"action": "mix OBJ", "description": "chemically mix a container"}, {"action": "go to LOC", "description": "move to a new location"}, {"action": "eat OBJ", "description": "eat a food"}, {"action": "flush OBJ", "description": "flush a toilet"}, {"action": "focus on OBJ", "description": "signal intent on a task object"}, {"action": "wait", "description": "take no action for 10 iterations"}, {"action": "wait1", "description": "take no action for 1 iteration"}, {"action": "task", "description": "describe current task"}, {"action": "inventory", "description": "list your inventory"}, {"action": "0", "description": "choose action 0"}, {"action": "1", "description": "choose action 1"}, {"action": "2", "description": "choose action 2"}, {"action": "3", "description": "choose action 3"}, {"action": "4", "description": "choose action 4"}, {"action": "5", "description": "choose action 5"}]. 
Replace the placeholder OBJ/LOC in your action with the specific object or location name. For example, use actions like "go to kitchen", "open cupboard" or "pick up orange". 
""".strip()

INIT_WEBSHOP = """
You are web shopping. I will give you instructions about what to do. You have to follow the instructions. Every round I will give you an observation and a list of available actions, you have to respond an action based on the state and instruction.
You can use search action if search is available. You can click one of the buttons in clickables. An action should be of the following structure:
search[keywords]
click[value]

Remember: 
- If your action is not valid, the web page will not change and nothing will happen.
- Keywords in search are up to you, but the value in click must be a value in the list of available actions. Your keywords in search should be carefully designed.
- You need to take steps progressively to meet all requirements. It is unlikely that a single action will fulfill the task immediately. For example, if the instruction is "Find me men's t-shirts with fit type: youth and size: large", you will need to: search first, then, click[large], next, click[youth], finllay, click[Buy Now].
""".strip()


INIT_ALFWORLD = """
Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. For each of your turn, you will be given a list of actions which you can choose one to perform in this turn. 
If you choose 'ACTION', you should directly output the action in this turn. Your output must strictly follow this format: 'Action: your next action'. After each turn, the environment will give you immediate feedback based on which you plan your next few steps. If the environment outputs 'Nothing happened', that means the previous action is invalid and you should try more options. 
Reminder: the action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal. Think when necessary, try to act directly more in the process.
""".strip()


INIT_TEXTCRAFT = """
You are given a few useful crafting recipes to craft items in Minecraft. Every round I will give you an observation, you have to respond to an action based on the state and instruction. 
Here are the valid actions you can take (no other formats will be accepted):
get [object]: "get" an object (ingredients) from the inventory or the environment(e.g., get 1 red dye)
inventory: look up the game "inventory" by inventory
craft [target object] using [input ingredients]: "craft" (target) using any of the crafting commands(e.g., craft 1 red dye using 1 beetroot). 
You can use **ONLY** these crafting commands provided, do not use your own crafting commands. However, if the crafting command uses a generic ingredient like "planks", you can use special types of the same ingredient e.g. dark oak “planks” in the command instead. 
""".strip()


INIT_BABYAI = """
You are an exploration master that wants to finish every goal you are given. Every round I will give you an observation, and you have to respond an action and your thought based on the observation to finish the given task. You are placed in a room and you need to accomplish the given goal with actions. You can use the following actions: 
- turn right
- turn left 
- move forward 
- go to <obj> <id> 
- pick up <obj> <id> 
- go through <door> <id>: <door> must be an open door. 
- toggle and go through <door> <id>: <door> can be a closed door or a locked door. If you want to open a locked door, you need to carry a key that is of the same color as the locked door. 
- toggle: there is a closed or locked door right in front of you and you can toggle it.
""".strip()


THINK_MODE_0 = """
Your response should use the following format:
Thought: your thoughts
Action: your next action
"""

THINK_MODE_BASE = """
[Output Format]
At each step, if you need to think, use the following format:
<|begin_of_thinking|>
Thought: your_thoughts
<|end_of_thinking|>
<>begin_of_answer|>
Action: your_next_action
<|end_of_answer|>

If you do not need to think at current step, use the following format:
<|begin_of_answer|>
Action: your_next_action
<|end_of_answer|>
""".strip()

'''
THINK_MODE_1 = """
At each step, you must respond immediately with the next one step action.

[Output Format]
Your output must adhere to the following format:
<|begin_of_answer|> 
Action: your_next_action
<|end_of_answer|>
""".strip()
'''

THINK_MODE_1 = """
At each step, you must respond immediately with the next one step action.

[Output Format]
Your output must adhere to the following format:
<action>your_next_action</action>
""".strip()

THINK_MODE_2 = """
At each step, you must first analyze the current state, avaliable actions, then generate next step action.

[Output Format]
Your output must adhere to the following format:
<|begin_of_thinking|>
Current state: [Current state and inventory]
Available actions: [Valid actions you can take at this step]
Response: [Give a preliminary response]
<|end_of_thinking|>
<|begin_of_answer|> 
Action: your_next_action
<|end_of_answer|>
""".strip()


THINK_MODE_3 = """
At each step, you must reflect on previous actions, adapt your strategy if necessary, and then determine the next action.

[Output Format]
Your output must adhere to the following format:
<|begin_of_thinking|>
Goal: [What needs to be accomplished]
Reflection: [Assess the effectiveness of recent actions and summarize key insights]
Response: [Give a preliminary response]
<|end_of_thinking|>
<|begin_of_answer|> 
Action: your_next_action
<|end_of_answer|>
""".strip()

THINK_MODE_4 = """
At each step, you must reflect on the current goal, list the possible candidate actions, evaluate them, and then decide on the next action.

[Output Format]
Your output must strictly follow the format below:
<|begin_of_thinking|>
Goal: [State the objective to be achieved]
Candidate actions: [List the valid actions at this step]
Evaluation: [Assess the potential effectiveness of each candidate action]
Response: [Select and justify your preliminary decision]
<|end_of_thinking|>
<|begin_of_answer|>
Action: your_next_action
<|end_of_answer|>
""".strip()

'''
THINK_MODE_5 = """
There are four thinking levels:
Level 1 - Instinctive Response: Immediate reaction based on intuition, no analysis.
Level 2 - Situational Awareness: Assess current state and available actions before acting.
Level 3 - Experience Integration: Reflect on past actions and outcomes to inform current decisions.
Level 4 - Strategic Planning: Assess the task goal, past lessons, and current state to analyze the future impact of each candidate action and optimize the decision.

At each step, you may choose an appropriate level of thinking (one of the four levels) to respond based on the given scenario.

[Output Format] 
Your output must adhere to the following format:
EXAMPLE 1:
Thinking Level: 1
<|begin_of_answer|> 
Action: your_next_action
<|end_of_answer|>

EXAMPLE 2:
Thinking Level: 2
<|begin_of_thinking|>
Current state: [Current state and inventory]
Available actions: [What actions are valid right now]
Response: [Give a preliminary response]
<|end_of_thinking|>
<|begin_of_answer|> 
Action: your_next_action
<|end_of_answer|>

EXAMPLE 3:
Thinking Level: 3
<|begin_of_thinking|>
Goal: [What needs to be accomplished]
Current state: [Current state and inventory]
Available actions: [What actions are valid right now]
Reflection: [How effective were recent actions, what was learned]
Response: [Give a preliminary response]
<|end_of_thinking|>
<|begin_of_answer|> 
Action: your_next_action
<|end_of_answer|>

EXAMPLE 4:
Thinking Level: 4
<|begin_of_thinking|>
Goal: [What needs to be accomplished]
Current state: [Current state and inventory]
Available actions: [What actions are valid right now]
Reflection: [How effective were recent actions, what was learned]
Evaluation: [Assess the potential effectiveness of each candidate action]
Response: [Give a preliminary response]
<|end_of_thinking|>
<|begin_of_answer|>
Action: your_next_action
<|end_of_answer|>
""".strip()
'''


THINK_MODE_5 ="""
There are four thinking levels:
Level 1 - Instinctive Response: Immediate reaction based on intuition, no analysis.
Level 2 - Situational Awareness: Assess current state and available actions before acting.
Level 3 - Experience Integration: Reflect on past actions and outcomes to inform current decisions.
Level 4 - Strategic Planning: Assess the task goal, past lessons, and current state to analyze the future impact of each candidate action and optimize the decision.

At each step, you must first choose an appropriate level of thinking (one of the four levels) to respond based on the given scenario. The chosen level MUST be enclosed within <level> </level> tags.
Next, reason through the problem step-by-step using the chosen thinking level. This reasoning process MUST be enclosed within <think> </think> tags. For Level 1 (Instinctive Response), use the fixed text: "Okay, I think I have finished thinking." For Levels 2-4, provide detailed reasoning as shown in examples.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.

[Output Format] 
Your output must adhere to the following format:
EXAMPLE 1:
<level>1</level>
<think>Okay, I think I have finished thinking.</think>
<action>your_next_action</action>

EXAMPLE 2:
<level>2</level>
<think>
Current state: [Current state and inventory]
Available actions: [What actions are valid right now]
Reasoning: [Choose the best action and explain why]
</think>
<action>your_next_action</action>

EXAMPLE 3:
<level>3</level>
<think>
Goal: [What needs to be accomplished]
Current state: [Current state and inventory]
Available actions: [What actions are valid right now]
Reflection: [How effective were recent actions, what was learned]
Reasoning: [Choose the best action based on experience]
</think>
<action>your_next_action</action>

EXAMPLE 4:
<level>4</level>
<think>
Goal: [What needs to be accomplished]
Current state: [Current state and inventory]
Available actions: [What actions are valid right now]
Reflection: [How effective were recent actions, what was learned]
Evaluation: [Assess the potential effectiveness of each candidate action]
Reasoning: [Choose the optimal action with strategic reasoning]
</think>
<action>your_next_action</action>
""".strip()


INIT_WEBARENA = """
You are an autonomous intelligent agent tasked with navigating a web browser. You will be given web-based tasks. These tasks will be accomplished through the use of specific actions you can issue.

Here's the information you'll have:
The user's objective: This is the task you're trying to complete.
The current web page's accessibility tree: This is a simplified representation of the webpage, providing key information.
The current web page's URL: This is the page you're currently navigating.
The open tabs: These are the tabs you have open.
The previous action: This is the action you just performed. It may be helpful to track your progress.

The actions you can perform fall into several categories:

Page Operation Actions:
`click [id]`: This action clicks on an element with a specific id on the webpage.
`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id. By default, the "Enter" key is pressed after typing unless press_enter_after is set to 0.
`hover [id]`: Hover over an element with id.
`press [key_comb]`:  Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v).
`scroll [direction=down|up]`: Scroll the page up or down.

Tab Management Actions:
`new_tab`: Open a new, empty browser tab.
`tab_focus [tab_index]`: Switch the browser's focus to a specific tab using its index.
`close_tab`: Close the currently active tab.

URL Navigation Actions:
`goto [url]`: Navigate to a specific URL.
`go_back`: Navigate to the previously viewed page.
`go_forward`: Navigate to the next page (if a previous 'go_back' action was performed).

Completion Action:
`stop [answer]`: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete, provide the answer as "N/A" in the bracket.

Homepage:
If you want to visit other websites, check out the homepage at http://homepage.com. It has a list of websites you can visit.
http://homepage.com/password.html lists all the account name and password for the websites. You can use them to log in to the websites.

To be successful, it is very important to follow the following rules:
1. You should only issue an action that is valid given the current observation
2. You should only issue one action at a time.
3. You should follow the examples to reason step by step and then issue the next action.
4. Generate the action in the correct format. Start with a "In summary, the next action I will perform is" phrase, followed by action inside ``````. For example, "In summary, the next action I will perform is ```click [1234]```".
5. Issue stop action when you think you have achieved the objective. Don't generate anything after stop.
""".strip()


SCI_RULE = """
4. **Key Restrictions:**
- The agent **CANNOT** 'go to bathroom' or 'open door to bathroom' at hallway !!!!!!
- The agent **CANNOT** go to the following rooms at this time: {invalid_rooms} !!!!!!
""".strip()

ACTION_SCIWORLD = """
#### Allowed Action Types
1. The generated action must be a valid combination of a verb and an object from the lists below (replace the *OBJ* in verbs with valid objects) or a number in choose actions.   
- Possible **Verbs**: {possible_actions}
- Possible Objects **OBJ**: {possible_objects}, {inventory}
- Possible choose actions: 0, 1, 2, 3, 4, 5
2. The combination must be logical. Here are **invalid combinations**: "go to air", "pick up air", "open air", "focus on air", "focus on agent", "focus on [room]".
3. Always consider the correct sequence of rooms and actions. For example: 
- To measure the temperature of unknown substance B which is located in bathroom, since **bathroom** is inside the **kitchen**, you **MUST** start with 'go to kitchen' first. Then 'look round', if **thermometer is in kitchen**, first 'focus on thermometer', then 'pick up thermometer' at kitchen, finally 'go to bathroom.', 'focus on unknown substance B' at bathroom. 
- To find (an) living thing and move it to the blue box in living room, you **MUST** start with find the living thing first(more possible in 'outside' or 'greenhouse'), do not go to living room directly.
""".strip()


ACTION_WEBSHOP = """
#### Allowed Action Types
The agent is **restricted** to performing actions in the following list:
- **search[KEYWORDS]**: perform a web search
- **click[OBJ]**: click an object or link on a webpage
""".strip()


ACTION_TEXTCRAFT = """
#### Allowed Action Types
The agent is **restricted** to performing actions in the following list:
- **get [NUMBER] [OBJ]**: "get" the specified number of objects.
- **inventory**: look up the game "inventory"
- **craft [TARGET] using [INGREDIENTS]**: "craft" the target item using any of the crafting commands.

NOTE:
- **Do not assume** the agent has all the necessary ingredients unless they have explicitly checked the inventory or obtained them.
- Items must be obtained in the correct sequence for crafting. For example, to craft 1 iron ingot, the agent must first get 9 iron nuggets. The agent cannot directly obtain 1 iron ingot.
- **Crafting is only possible** once the agent has gathered all required ingredients, including both the correct **quantity** and **types**.
""".strip()

ACTION_ALFWORLD = """
#### Allowed Action Types
The agent's next step action **MUST** be selected from the list of valid actions provided above **valid_actions**. 

{valid_actions}

Any candidate action **NOT** present in this list is considered **non-executable** and **MUST NOT** be performed.
""".strip()

ACTION_WEBARENA = """
The actions you can perform fall into several categories:

Page Operation Actions:
`click [id]`: This action clicks on an element with a specific id on the webpage.
`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id. By default, the "Enter" key is pressed after typing unless press_enter_after is set to 0.
`hover [id]`: Hover over an element with id.
`press [key_comb]`:  Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v).
`scroll [direction=down|up]`: Scroll the page up or down.

Tab Management Actions:
`new_tab`: Open a new, empty browser tab.
`tab_focus [tab_index]`: Switch the browser's focus to a specific tab using its index.
`close_tab`: Close the currently active tab.

URL Navigation Actions:
`goto [url]`: Navigate to a specific URL.
`go_back`: Navigate to the previously viewed page.
`go_forward`: Navigate to the next page (if a previous 'go_back' action was performed).

Completion Action:
`stop [answer]`: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete, provide the answer as "N/A" in the bracket.

Homepage:
If you want to visit other websites, check out the homepage at http://homepage.com. It has a list of websites you can visit.
http://homepage.com/password.html lists all the account name and password for the websites. You can use them to log in to the websites.

To be successful, it is very important to follow the following rules:
1. You should only issue an action that is valid given the current observation
2. You should only issue one action at a time.
3. You should follow the examples to reason step by step and then issue the next action.
4. Generate the action in the correct format. Start with a "In summary, the next action I will perform is" phrase, followed by action inside ``````. For example, "In summary, the next action I will perform is ```click [1234]```".
5. Issue stop action when you think you have achieved the objective. Don't generate anything after stop.
""".strip()

ACTION_BABYAI = """
#### Allowed Action Types
You can use the following actions:

{valid_actions}
""".strip()

INSTRUCT_GOLD_ACTION_CRITIC = """
Your task is to critique the candidate next-step action based on the agent's task goal and interaction history. The gold next-step action is provided as a reference but should not be explicitly mentioned in your critique.

{available_actions}

### Critique Steps

#### Step 1: Analyze Candidate Action
Examine the candidate's action based on the following criteria, then assign an overall grade using this scale: **Excellent, Good, Neutral, Poor, Very Poor**.

### Critique Dimensions
* **Contribution**: Assess whether the action contributes to completing the agent's task. This includes both direct actions (e.g., picking up the target OBJ) and indirect actions (e.g., reasonable exploration that can provide additional environmental information and facilitate future progress).
* **Feasibility**: Assess whether the action is valid according to the agent's predefined ** Allowed Action Types** list.
* **Efficiency**: Analyze whether the action optimally achieves the task without unnecessary steps or redundancy.

#### Step 2: Provide Revision Suggestions
Suggest a modification to align the candidate's action better with the task or the agent's action capabilities. For example, if the action is not allowed, recommend an alternative from the action list that aligns better with the task goal.

### Critique Format
Please structure your critique in the following format:
## Contribution: [Analysis of Contribution].
## Feasibility: [Analysis of feasibility].
## Efficiency: [Analysis of efficiency].
## Overall Grading: [Overall grade: Excellent/Good/Neutral/Poor/Very Poor].
## Suggested Revision: [Brief revision suggestion, if applicable].

Note: Do not mention the gold action in any way in your critiques (e.g., “This action aligns with the gold action”).

### Inputs

{command}

The Agent's Task Goal and Interaction History:

{history}

Gold Next Step Action: {gold_action}

Candidate Next Step Action: {candidate_action}

Now, please provide your critique:
""".strip()


INSTRUCT_CRITIC = """
Your task is to critique the candidate's next-step action based on the agent's task goal and interaction history.

{available_actions}

### Critique Steps

#### Step 1: Analyze Candidate Action
Examine the candidate's action based on the following criteria and assign an overall grade using this scale: **Excellent, Good, Neutral, Poor, Very Poor**.

### Critique Dimensions
* **Contribution**: Assess whether the action contributes to completing the agent's task. This includes both direct actions (e.g., picking up the target OBJ) and indirect actions (e.g., reasonable exploration that can provide additional environmental information and facilitate future progress).
* **Feasibility**: Assess whether the action is valid according to the agent's predefined ** Allowed Action Types** list.
* **Efficiency**: Analyze whether the action optimally achieves the task without unnecessary steps or redundancy.

#### Step 2: Provide Revision Suggestions
Suggest a modification to align the candidate's action better with the task or the agent's action capabilities. Note that the suggested revision should be based on the **Allowed Action and Object Types**.

### Critique Format
Please structure your critique in the following format:
## Contribution: [Analysis of Contribution].
## Feasibility: [Analysis of feasibility].
## Efficiency: [Analysis of efficiency].
## Overall Grading: [Overall grade: Excellent/Good/Neutral/Poor/Very Poor].
## Suggested Revision: [Brief revision suggestion, if applicable].

### Inputs
{command}

The Agent's Task Goal and Interaction History:

{history}

Candidate Next Step Action: {candidate_action}

Now, please provide your critique:
""".strip()


INSTRUCT_GOLD_PATH_CRITIC = """
Your task is to critique the candidate next-step action based on the agent's task goal and interaction history. The gold path for current task is provided as a reference to guide your critique, but do not explicitly mention it in your critique.

{available_actions}

### Critique Steps

#### Step 1: Analyze Candidate Action
Examine the candidate's action based on the following criteria, then assign an overall grade using this scale: **Excellent, Good, Neutral, Poor, Very Poor**.

### Critique Dimensions
* **Contribution**: Assess whether the action contributes to completing the agent's task. This includes both direct actions (e.g., picking up the target OBJ) and indirect actions (e.g., reasonable exploration that can provide additional environmental information and facilitate future progress).
* **Feasibility**: Assess whether the action is valid according to the agent's predefined ** Allowed Action Types** list.
* **Efficiency**: Analyze whether the action optimally achieves the task without unnecessary steps or redundancy.

#### Step 2: Provide Revision Suggestions
Suggest a modification to align the candidate's action better with the task or the agent's action capabilities. For example, if the action is not allowed, recommend an alternative from the action list that aligns better with the task goal.

### Critique Format
Please structure your critique in the following format:
## Contribution: [Analysis of Contribution].
## Feasibility: [Analysis of feasibility].
## Efficiency: [Analysis of efficiency].
## Overall Grading: [Overall grade: Excellent/Good/Neutral/Poor/Very Poor].
## Suggested Revision: [Brief revision suggestion, if applicable].

Note: Do not mention the gold action in any way in your critiques (e.g., “This action aligns with the gold action”).

### Referenced Gold Path for Current Task

{gold_path}

### Inputs

{command}

The Agent's Task Goal and Interaction History:

{history}

Candidate Next Step Action: {candidate_action}

Now, please provide your critique:

""".strip()

INSTRUCT_PARSE = """
Your task is to select the most appropriate action from the list of valid actions below and output it. If the provided action is already valid, simply output it as is.

Please ensure that the verb and object in your output are logically compatible. The object *MUST* be from the Possible Object List. *DO NOT* add parentheses after the object!

Valid Actions:

{available_actions_string}

For example:
Provided action: 
go kitchen
Refined action: 
go to kitchen

Provided action:
move baby wolf in inventory to blue box
Refined action:
move baby wolf to blue box

Provided action:
move a painting of two people playing. The artist is listed as Bob to orange box
Refined action:
move painting to orange box

Provided action:
use thermometer in inventory on unknown substance B
Refined action:
use thermometer on unknown substance

Now, it's your turn to answer:  
Provided action: 
{action}
Refined action:
""".strip()


INSTRUCT_REVISE_THOUGHT = """
Your task is to transform the agent's reasoning process from a critique-based approach to a first-person, self-driven reasoning style (e.g., "I think/realize/understand/find/believe that ..., so ..."), while preserving the original intention.
Remember: 
- The revised thought MUST BE **brief**, and logically sound.
- Avoid referring to external critiques (e.g., "Based on the critiques", "The critiques suggest...").
- If mentioning inefficiency or issues (e.g., "the previous action is inefficient"), explicitly identify the specific action. If the specific action cannot be determined, omit such statements entirely.

For example:
Original Thought:
Based on the critiques, I understand that repeatedly using the thermometer without taking other necessary actions to increase the water's temperature does not directly advance the task of boiling the water. The critiques suggest that waiting for a period to allow the water to heat up further is a more efficient and effective approach. This action aligns well with the task goal of boiling water and allows the water to reach its boiling point without unnecessary actions.
Revised Thought:
I realize that frequently checking the thermometer without implementing additional measures to raise the water's temperature is not efficient for boiling water. I can 'wait1' and allow the water to heat up naturally.
Original Thought:
The critiques provide valuable insights on the potential next steps. I understand that the current action is inefficient and premature, as it attempts to move the flower pot containing the cherry tree to the green box without first picking it up. The critiques suggest that I should pick up the flower pot first to add it to my inventory before attempting to move it to the green box. This is a necessary step to align with the task goal of moving the focused plant to the green box in the living room.
Revised Thought:
I find that trying to move the flower pot to the green box without picking it up first is inefficient. I should pick up the flower pot to add it to my inventory before moving it to the green box.
Original Thought:
Based on the critiques, I understand that my previous actions were inefficient and not feasible in the current context. The suggested revisions emphasize the importance of moving to the hallway before proceeding to the living room. I will adjust my strategy to follow this revised approach.
Revised Thought:
I believe that I should first move to the hallway before proceeding to the living room.
Original Thought:
I have received the critique and I understand that my previous actions were not aligned with the task goal. The critique suggests that I should focus on the "baby baby elephant" instead, as it has the longest life span among the animals present outside. I will take the suggested revision into consideration and adjust my strategy accordingly.
Revised Thought:
I think that I have to focus on the "baby baby elephant", because it has the longest lifespan among the animals outside.

Now it is your turn to revise:

Original Thought:
{thought}
Revised Thought:
""".strip()
'''

INSTRUCT_REVISE_THOUGHT = """
Your task is to transform the agent's reasoning process from a critique-based approach to one centered on self-reflection(e.g., "I think/understand it is ...", "Reflect on my previous action, ..."), while preserving the original intention. Ensure the revised thought is fluent, cohesive, and logically sound, and avoids terms such as "based on the critique."

For example:
Original Thought:
Based on the critiques, I understand that repeatedly using the thermometer without taking other necessary actions to increase the water's temperature does not directly advance the task of boiling the water. The critiques suggest that waiting for a period to allow the water to heat up further is a more efficient and effective approach. This action aligns well with the task goal of boiling water and allows the water to reach its boiling point without unnecessary actions.
Revised Thought:
I realize that frequently checking the thermometer without implementing additional measures to raise the water's temperature isn't effectively moving me closer to the goal of boiling the water. By pausing and allowing the water time to heat up naturally, I can adopt a more efficient and effective strategy. This approach aligns with the objective of boiling the water, enabling it to reach its boiling point without engaging in unnecessary actions.

Now it is your turn to revise:

Original Thought:
{thought}
Revised Thought:
""".strip()
'''

INSTRUCT_REFLEXION = """
Your task is to provide a next plan of action based on the previous attempt's interaction history. The next plan should reflect the lessons learned from the failure and adjust for better results.

Previous Trial:
{history}

Next Plan:
""".strip()


INSTRUCT_GENERATE_ADAPTIVE_THINKING = """Based on the following information, generate a thinking process that leads to the gold action.

Task Description: {task_description}

Current State/Observation: {current_obs}

History of Previous Actions and Observations:
{history}

Next Gold Action (the correct action to take): {gold_action}

You need to choose an appropriate thinking level based on the complexity of the current situation:
- Level 1 - Instinctive Response: For simple, obvious actions that require no analysis
- Level 2 - Situational Awareness: When you need to assess the current state and available actions
- Level 3 - Experience Integration: When past actions and their outcomes should inform the decision
- Level 4 - Strategic Planning: For complex decisions requiring analysis of future impacts

Generate a thinking process following the exact format for your chosen level that naturally leads to taking the gold action.

For Level 1:
Thinking Level: 1
<|begin_of_answer|> 
Action: {gold_action}
<|end_of_answer|>

For Level 2:
Thinking Level: 2
<|begin_of_thinking|>
Current state: [Describe current state based on observation]
Available actions: [What actions seem reasonable in this state]
Response: [Your reasoning that leads to the gold action]
<|end_of_thinking|>
<|begin_of_answer|> 
Action: {gold_action}
<|end_of_answer|>

For Level 3:
Thinking Level: 3
<|begin_of_thinking|>
Goal: [State the goal from task description]
Current state: [Describe current state based on observation]
Available actions: [What actions seem reasonable]
Reflection: [Analyze recent actions from history and what was learned]
Response: [Your reasoning that leads to the gold action]
<|end_of_thinking|>
<|begin_of_answer|> 
Action: {gold_action}
<|end_of_answer|>

For Level 4:
Thinking Level: 4
<|begin_of_thinking|>
Goal: [State the goal from task description]
Current state: [Describe current state based on observation]
Available actions: [What actions seem reasonable]
Reflection: [Analyze recent actions from history and what was learned]
Evaluation: [Assess why the gold action would be most effective]
Response: [Your strategic reasoning that leads to the gold action]
<|end_of_thinking|>
<|begin_of_answer|>
Action: {gold_action}
<|end_of_answer|>
"""

INSTRUCT_GENERATE_LEVEL_4_THINKING = """
Based on the following information, generate a thinking process that leads to the gold action.

Task Description: {task_description}

Current State/Observation: {current_obs}

History of Previous Actions and Observations:
{history}

Next Gold Action (the correct action to take): {gold_action}

You need to generate a thinking process following the exact format:

<think>
Goal: [State the goal from task description]
Current state: [Describe current state based on observation]
Available actions: [What actions seem reasonable]
Reflection: [Analyze recent actions from history and what was learned]
Evaluation: [Assess why the gold action would be most effective]
Response: [Your strategic reasoning that leads to the gold action]
</think>
"""

INSTRUCT_GENERATE_LEVEL_3_THINKING = """
Based on the following information, generate a thinking process that leads to the gold action.

Task Description: {task_description}

Current State/Observation: {current_obs}

History of Previous Actions and Observations:
{history}

Next Gold Action (the correct action to take): {gold_action}

You need to generate a thinking process following the exact format:

<think>
Goal: [State the goal from task description]
Current state: [Describe current state based on observation]
Available actions: [What actions seem reasonable]
Reflection: [Analyze recent actions from history and what was learned]
Response: [Your reasoning that leads to the gold action]
</think>
"""

INSTRUCT_GENERATE_LEVEL_2_THINKING = """
Based on the following information, generate a thinking process that leads to the gold action.

Task Description: {task_description}

Current State/Observation: {current_obs}

History of Previous Actions and Observations:
{history}

Next Gold Action (the correct action to take): {gold_action}

You need to generate a thinking process following the exact format:

<think>
Goal: [State the goal from task description]
Current state: [Describe current state based on observation]
Available actions: [What actions seem reasonable]
Response: [Your reasoning that leads to the gold action]
</think>
"""


TEMPLATE_NO_HIS_ADA = """
{system_prompt}

{task_description}

Now it's your turn to generate next step response.
""".strip()


TEMPLATE_ADA = """
{system_prompt}

{task_description}

Here is the task history for past 10 steps:
{action_history}
Observation: {current_observation}

Now it's your turn to generate next step response.
""".strip()


SCIWORLD_TEMPLATE_NO_HIS_ADA = """
You are an agent for science world. Every round I will give you an observation, you have to respond an action based on the observation to finish the given task. 
Your current task is: {task_description}

Your current observation is: {current_observation}

Here are the actions you may take: 
[
{{"action": "open OBJ", "description": "open a container"}},
{{"action": "close OBJ", "description": "close a container"}},
{{"action": "activate OBJ", "description": "activate a device"}},
{{"action": "deactivate OBJ", "description": "deactivate a device"}},
{{"action": "connect OBJ to OBJ", "description": "connect electrical components"}},
{{"action": "disconnect OBJ", "description": "disconnect electrical components"}},
{{"action": "use OBJ [on OBJ]", "description": "use a device/item"}},
{{"action": "look around", "description": "describe the current room"}},
{{"action": "look at OBJ", "description": "describe an object in detail"}},
{{"action": "look in OBJ", "description": "describe a container's contents"}},
{{"action": "read OBJ", "description": "read a note or book"}},
{{"action": "move OBJ to OBJ", "description": "move an object to a container"}},
{{"action": "pick up OBJ", "description": "move an object to the inventory"}},
{{"action": "put down OBJ", "description": "drop an inventory item"}},
{{"action": "pour OBJ into OBJ", "description": "pour a liquid into a container"}},
{{"action": "dunk OBJ into OBJ", "description": "dunk a container into a liquid"}},
{{"action": "mix OBJ", "description": "chemically mix a container"}},
{{"action": "go to LOC", "description": "move to a new location"}},
{{"action": "eat OBJ", "description": "eat a food"}},
{{"action": "flush OBJ", "description": "flush a toilet"}},
{{"action": "focus on OBJ", "description": "signal intent on a task object"}},
{{"action": "wait", "description": "take no action for 10 iterations"}},
{{"action": "wait1", "description": "take no action for 1 iteration"}},
{{"action": "task", "description": "describe current task"}},
{{"action": "inventory", "description": "list your inventory"}}
]
Replace the placeholder OBJ/LOC in your action with the specific object or location name. For example, use actions like "go to kitchen", "open cupboard" or "pick up orange". 

Now it's your turn to generate next step response.

There are four thinking levels:
Level 1 - Instinctive Response: Immediate reaction based on intuition, no analysis.
Level 2 - Situational Awareness: Assess current state and available actions before acting.
Level 3 - Experience Integration: Reflect on past actions and outcomes to inform current decisions.
Level 4 - Strategic Planning: Assess the task goal, past lessons, and current state to analyze the future impact of each candidate action and optimize the decision.

At each step, you must first choose an appropriate level of thinking (one of the four levels) to respond based on the given scenario. The chosen level MUST be enclosed within <level> </level> tags.
Next, reason step-by-step using the chosen thinking level. This reasoning process MUST be enclosed within <think> </think> tags. For Level 1 (Instinctive Response), use the fixed text: "Okay, I think I have finished thinking." For Levels 2-4, provide detailed reasoning as shown in examples.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.

[Output Format] 
Your output must adhere to the following format:
EXAMPLE 1:
<level>1</level>
<think>Okay, I think I have finished thinking.</think>
<action>your_next_action</action>

EXAMPLE 2:
<level>2</level>
<think>
Current state: [Current state and inventory]
Available actions: [What actions are valid right now]
Reasoning: [Choose the best action and explain why]
</think>
<action>your_next_action</action>

EXAMPLE 3:
<level>3</level>
<think>
Goal: [What needs to be accomplished]
Current state: [Current state and inventory]
Available actions: [What actions are valid right now]
Reflection: [How effective were recent actions, what was learned]
Reasoning: [Choose the best action based on experience]
</think>
<action>your_next_action</action>

EXAMPLE 4:
<level>4</level>
<think>
Goal: [What needs to be accomplished]
Current state: [Current state and inventory]
Available actions: [What actions are valid right now]
Reflection: [How effective were recent actions, what was learned]
Evaluation: [Assess the potential effectiveness of each candidate action]
Reasoning: [Choose the optimal action with strategic reasoning]
</think>
<action>your_next_action</action>
""".strip()


SCIWORLD_TEMPLATE_ADA = """
You are an agent for science world. Every round I will give you an observation, you have to respond an action based on the observation to finish the given task. 
Your current task is: {task_description}

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: 
{action_history}
You are now at step {current_step} and your current observation is: {current_observation}

Here are the actions you may take:

[
{{"action": "open OBJ", "description": "open a container"}},
{{"action": "close OBJ", "description": "close a container"}},
{{"action": "activate OBJ", "description": "activate a device"}},
{{"action": "deactivate OBJ", "description": "deactivate a device"}},
{{"action": "connect OBJ to OBJ", "description": "connect electrical components"}},
{{"action": "disconnect OBJ", "description": "disconnect electrical components"}},
{{"action": "use OBJ [on OBJ]", "description": "use a device/item"}},
{{"action": "look around", "description": "describe the current room"}},
{{"action": "look at OBJ", "description": "describe an object in detail"}},
{{"action": "look in OBJ", "description": "describe a container's contents"}},
{{"action": "read OBJ", "description": "read a note or book"}},
{{"action": "move OBJ to OBJ", "description": "move an object to a container"}},
{{"action": "pick up OBJ", "description": "move an object to the inventory"}},
{{"action": "put down OBJ", "description": "drop an inventory item"}},
{{"action": "pour OBJ into OBJ", "description": "pour a liquid into a container"}},
{{"action": "dunk OBJ into OBJ", "description": "dunk a container into a liquid"}},
{{"action": "mix OBJ", "description": "chemically mix a container"}},
{{"action": "go to LOC", "description": "move to a new location"}},
{{"action": "eat OBJ", "description": "eat a food"}},
{{"action": "flush OBJ", "description": "flush a toilet"}},
{{"action": "focus on OBJ", "description": "signal intent on a task object"}},
{{"action": "wait", "description": "take no action for 10 iterations"}},
{{"action": "wait1", "description": "take no action for 1 iteration"}},
{{"action": "task", "description": "describe current task"}},
{{"action": "inventory", "description": "list your inventory"}}
]
Replace the placeholder OBJ/LOC in your action with the specific object or location name. For example, use actions like "go to kitchen", "open cupboard" or "pick up orange". 

Now it's your turn to generate next step response. 

There are four thinking levels:
Level 1 - Instinctive Response: Immediate reaction based on intuition, no analysis.
Level 2 - Situational Awareness: Assess current state and available actions before acting.
Level 3 - Experience Integration: Reflect on past actions and outcomes to inform current decisions.
Level 4 - Strategic Planning: Assess the task goal, past lessons, and current state to analyze the future impact of each candidate action and optimize the decision.

You must first choose an appropriate level of thinking (one of the four levels) to respond based on the given scenario. The chosen level MUST be enclosed within <level> </level> tags.
Next, reason step-by-step using the chosen thinking level. This reasoning process MUST be enclosed within <think> </think> tags. For Level 1 (Instinctive Response), use the fixed text: "Okay, I think I have finished thinking." For Levels 2-4, provide detailed reasoning as shown in examples.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.

[Output Format] 
Your output must adhere to the following format:
EXAMPLE 1:
<level>1</level>
<think>Okay, I think I have finished thinking.</think>
<action>your_next_action</action>

EXAMPLE 2:
<level>2</level>
<think>
Current state: [Current state and inventory]
Available actions: [What actions are valid right now]
Reasoning: [Choose the best action and explain why]
</think>
<action>your_next_action</action>

EXAMPLE 3:
<level>3</level>
<think>
Goal: [What needs to be accomplished]
Current state: [Current state and inventory]
Available actions: [What actions are valid right now]
Reflection: [How effective were recent actions, what was learned]
Reasoning: [Choose the best action based on experience]
</think>
<action>your_next_action</action>

EXAMPLE 4:
<level>4</level>
<think>
Goal: [What needs to be accomplished]
Current state: [Current state and inventory]
Available actions: [What actions are valid right now]
Reflection: [How effective were recent actions, what was learned]
Evaluation: [Assess the potential effectiveness of each candidate action]
Reasoning: [Choose the optimal action with strategic reasoning]
</think>
<action>your_next_action</action>
""".strip()



SCIWORLD_TEMPLATE_NO_HIS_ADA_V2 = """
You are an agent for science world. Every round I will give you an observation, you have to respond an action based on the observation to finish the given task.
Here are the actions you may take: 
[
{{"action": "open OBJ", "description": "open a container"}},
{{"action": "close OBJ", "description": "close a container"}},
{{"action": "activate OBJ", "description": "activate a device"}},
{{"action": "deactivate OBJ", "description": "deactivate a device"}},
{{"action": "connect OBJ to OBJ", "description": "connect electrical components"}},
{{"action": "disconnect OBJ", "description": "disconnect electrical components"}},
{{"action": "use OBJ [on OBJ]", "description": "use a device/item"}},
{{"action": "look around", "description": "describe the current room"}},
{{"action": "look at OBJ", "description": "describe an object in detail"}},
{{"action": "look in OBJ", "description": "describe a container's contents"}},
{{"action": "read OBJ", "description": "read a note or book"}},
{{"action": "move OBJ to OBJ", "description": "move an object to a container"}},
{{"action": "pick up OBJ", "description": "move an object to the inventory"}},
{{"action": "put down OBJ", "description": "drop an inventory item"}},
{{"action": "pour OBJ into OBJ", "description": "pour a liquid into a container"}},
{{"action": "dunk OBJ into OBJ", "description": "dunk a container into a liquid"}},
{{"action": "mix OBJ", "description": "chemically mix a container"}},
{{"action": "go to LOC", "description": "move to a new location"}},
{{"action": "eat OBJ", "description": "eat a food"}},
{{"action": "flush OBJ", "description": "flush a toilet"}},
{{"action": "focus on OBJ", "description": "signal intent on a task object"}},
{{"action": "wait", "description": "take no action for 10 iterations"}},
{{"action": "wait1", "description": "take no action for 1 iteration"}},
{{"action": "task", "description": "describe current task"}},
{{"action": "inventory", "description": "list your inventory"}}
]
Replace the placeholder OBJ/LOC in your action with the specific object or location name. For example, use actions like "go to kitchen", "open cupboard" or "pick up orange". 

There are four thinking levels:
Level 1 - Instinctive Response: Immediate reaction based on intuition, no analysis.
Level 2 - Situational Awareness: Assess current state and available actions before acting.
Level 3 - Experience Integration: Reflect on past actions and outcomes to inform current decisions.
Level 4 - Strategic Planning: Assess the task goal, past lessons, and current state to analyze the future impact of each candidate action and optimize the decision.

At each step, you must first choose an appropriate level of thinking (one of the four levels) to respond based on the given scenario. The chosen level MUST be enclosed within <level> </level> tags.
Next, reason step-by-step using the chosen thinking level. This reasoning process MUST be enclosed within <think> </think> tags. For Level 1 (Instinctive Response), use the fixed text: "Okay, I think I have finished thinking." For Levels 2-4, provide detailed reasoning as shown in examples.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.

[Output Format] 
Your output must adhere to the following format:
EXAMPLE 1:
<level>1</level>
<think>Okay, I think I have finished thinking.</think>
<action>your_next_action</action>

EXAMPLE 2:
<level>2</level>
<think>
Current state: [Current state and inventory]
Available actions: [What actions are valid right now]
Reasoning: [Choose the best action and explain why]
</think>
<action>your_next_action</action>

EXAMPLE 3:
<level>3</level>
<think>
Goal: [What needs to be accomplished]
Current state: [Current state and inventory]
Available actions: [What actions are valid right now]
Reflection: [How effective were recent actions, what was learned]
Reasoning: [Choose the best action based on experience]
</think>
<action>your_next_action</action>

EXAMPLE 4:
<level>4</level>
<think>
Goal: [What needs to be accomplished]
Current state: [Current state and inventory]
Available actions: [What actions are valid right now]
Reflection: [How effective were recent actions, what was learned]
Evaluation: [Assess the potential effectiveness of each candidate action]
Reasoning: [Choose the optimal action with strategic reasoning]
</think>
<action>your_next_action</action>

Your current task is: {task_description}
Your current observation is: {current_observation}
Now it's your turn to generate next step response.
""".strip()


SCIWORLD_TEMPLATE_ADA_V2 = """
You are an agent for science world. Every round I will give you an observation, you have to respond an action based on the observation to finish the given task. 
Here are the actions you may take:
[
{{"action": "open OBJ", "description": "open a container"}},
{{"action": "close OBJ", "description": "close a container"}},
{{"action": "activate OBJ", "description": "activate a device"}},
{{"action": "deactivate OBJ", "description": "deactivate a device"}},
{{"action": "connect OBJ to OBJ", "description": "connect electrical components"}},
{{"action": "disconnect OBJ", "description": "disconnect electrical components"}},
{{"action": "use OBJ [on OBJ]", "description": "use a device/item"}},
{{"action": "look around", "description": "describe the current room"}},
{{"action": "look at OBJ", "description": "describe an object in detail"}},
{{"action": "look in OBJ", "description": "describe a container's contents"}},
{{"action": "read OBJ", "description": "read a note or book"}},
{{"action": "move OBJ to OBJ", "description": "move an object to a container"}},
{{"action": "pick up OBJ", "description": "move an object to the inventory"}},
{{"action": "put down OBJ", "description": "drop an inventory item"}},
{{"action": "pour OBJ into OBJ", "description": "pour a liquid into a container"}},
{{"action": "dunk OBJ into OBJ", "description": "dunk a container into a liquid"}},
{{"action": "mix OBJ", "description": "chemically mix a container"}},
{{"action": "go to LOC", "description": "move to a new location"}},
{{"action": "eat OBJ", "description": "eat a food"}},
{{"action": "flush OBJ", "description": "flush a toilet"}},
{{"action": "focus on OBJ", "description": "signal intent on a task object"}},
{{"action": "wait", "description": "take no action for 10 iterations"}},
{{"action": "wait1", "description": "take no action for 1 iteration"}},
{{"action": "task", "description": "describe current task"}},
{{"action": "inventory", "description": "list your inventory"}}
]
Replace the placeholder OBJ/LOC in your action with the specific object or location name. For example, use actions like "go to kitchen", "open cupboard" or "pick up orange". 

There are four thinking levels:
Level 1 - Instinctive Response: Immediate reaction based on intuition, no analysis.
Level 2 - Situational Awareness: Assess current state and available actions before acting.
Level 3 - Experience Integration: Reflect on past actions and outcomes to inform current decisions.
Level 4 - Strategic Planning: Assess the task goal, past lessons, and current state to analyze the future impact of each candidate action and optimize the decision.

At each step, you must first choose an appropriate level of thinking (one of the four levels) to respond based on the given scenario. The chosen level MUST be enclosed within <level> </level> tags.
Next, reason step-by-step using the chosen thinking level. This reasoning process MUST be enclosed within <think> </think> tags. For Level 1 (Instinctive Response), use the fixed text: "Okay, I think I have finished thinking." For Levels 2-4, provide detailed reasoning as shown in examples.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.

[Output Format] 
Your output must adhere to the following format:
EXAMPLE 1:
<level>1</level>
<think>Okay, I think I have finished thinking.</think>
<action>your_next_action</action>

EXAMPLE 2:
<level>2</level>
<think>
Current state: [Current state and inventory]
Available actions: [What actions are valid right now]
Reasoning: [Choose the best action and explain why]
</think>
<action>your_next_action</action>

EXAMPLE 3:
<level>3</level>
<think>
Goal: [What needs to be accomplished]
Current state: [Current state and inventory]
Available actions: [What actions are valid right now]
Reflection: [How effective were recent actions, what was learned]
Reasoning: [Choose the best action based on experience]
</think>
<action>your_next_action</action>

EXAMPLE 4:
<level>4</level>
<think>
Goal: [What needs to be accomplished]
Current state: [Current state and inventory]
Available actions: [What actions are valid right now]
Reflection: [How effective were recent actions, what was learned]
Evaluation: [Assess the potential effectiveness of each candidate action]
Reasoning: [Choose the optimal action with strategic reasoning]
</think>
<action>your_next_action</action>

Your current task is: {task_description}

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: 
{action_history}
You are now at step {current_step} and your current observation is: {current_observation}

Now it's your turn to generate next step response. 
""".strip()


SCIWORLD_TEMPLATE_NO_HIS = """
You are an agent for science world. Every round I will give you an observation, you have to respond an action based on the observation to finish the given task.
Here are the actions you may take: 
[
{{"action": "open OBJ", "description": "open a container"}},
{{"action": "close OBJ", "description": "close a container"}},
{{"action": "activate OBJ", "description": "activate a device"}},
{{"action": "deactivate OBJ", "description": "deactivate a device"}},
{{"action": "connect OBJ to OBJ", "description": "connect electrical components"}},
{{"action": "disconnect OBJ", "description": "disconnect electrical components"}},
{{"action": "use OBJ [on OBJ]", "description": "use a device/item"}},
{{"action": "look around", "description": "describe the current room"}},
{{"action": "look at OBJ", "description": "describe an object in detail"}},
{{"action": "look in OBJ", "description": "describe a container's contents"}},
{{"action": "read OBJ", "description": "read a note or book"}},
{{"action": "move OBJ to OBJ", "description": "move an object to a container"}},
{{"action": "pick up OBJ", "description": "move an object to the inventory"}},
{{"action": "put down OBJ", "description": "drop an inventory item"}},
{{"action": "pour OBJ into OBJ", "description": "pour a liquid into a container"}},
{{"action": "dunk OBJ into OBJ", "description": "dunk a container into a liquid"}},
{{"action": "mix OBJ", "description": "chemically mix a container"}},
{{"action": "go to LOC", "description": "move to a new location"}},
{{"action": "eat OBJ", "description": "eat a food"}},
{{"action": "flush OBJ", "description": "flush a toilet"}},
{{"action": "focus on OBJ", "description": "signal intent on a task object"}},
{{"action": "wait", "description": "take no action for 10 iterations"}},
{{"action": "wait1", "description": "take no action for 1 iteration"}},
{{"action": "task", "description": "describe current task"}},
{{"action": "inventory", "description": "list your inventory"}}
]
Replace the placeholder OBJ/LOC in your action with the specific object or location name. For example, use actions like "go to kitchen", "open cupboard" or "pick up orange". 

Your response should use the following format:
Thought: your thoughts
Action: your next action

Your current task is: {task_description}
Your current observation is: {current_observation}
Now it's your turn to generate next step response.
""".strip()


SCIWORLD_TEMPLATE = """
You are an agent for science world. Every round I will give you an observation, you have to respond an action based on the observation to finish the given task. 
Here are the actions you may take:
[
{{"action": "open OBJ", "description": "open a container"}},
{{"action": "close OBJ", "description": "close a container"}},
{{"action": "activate OBJ", "description": "activate a device"}},
{{"action": "deactivate OBJ", "description": "deactivate a device"}},
{{"action": "connect OBJ to OBJ", "description": "connect electrical components"}},
{{"action": "disconnect OBJ", "description": "disconnect electrical components"}},
{{"action": "use OBJ [on OBJ]", "description": "use a device/item"}},
{{"action": "look around", "description": "describe the current room"}},
{{"action": "look at OBJ", "description": "describe an object in detail"}},
{{"action": "look in OBJ", "description": "describe a container's contents"}},
{{"action": "read OBJ", "description": "read a note or book"}},
{{"action": "move OBJ to OBJ", "description": "move an object to a container"}},
{{"action": "pick up OBJ", "description": "move an object to the inventory"}},
{{"action": "put down OBJ", "description": "drop an inventory item"}},
{{"action": "pour OBJ into OBJ", "description": "pour a liquid into a container"}},
{{"action": "dunk OBJ into OBJ", "description": "dunk a container into a liquid"}},
{{"action": "mix OBJ", "description": "chemically mix a container"}},
{{"action": "go to LOC", "description": "move to a new location"}},
{{"action": "eat OBJ", "description": "eat a food"}},
{{"action": "flush OBJ", "description": "flush a toilet"}},
{{"action": "focus on OBJ", "description": "signal intent on a task object"}},
{{"action": "wait", "description": "take no action for 10 iterations"}},
{{"action": "wait1", "description": "take no action for 1 iteration"}},
{{"action": "task", "description": "describe current task"}},
{{"action": "inventory", "description": "list your inventory"}}
]
Replace the placeholder OBJ/LOC in your action with the specific object or location name. For example, use actions like "go to kitchen", "open cupboard" or "pick up orange". 

Your response should use the following format:
Thought: your thoughts
Action: your next action

Your current task is: {task_description}

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: 
{action_history}
You are now at step {current_step} and your current observation is: {current_observation}

Now it's your turn to generate next step response. 
""".strip()


ALFWORLD_TEMPLATE_NO_HIS_ADA = """
Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. For each of your turn, you will be given a list of actions which you can choose one to perform in this turn. After each turn, the environment will give you immediate feedback based on which you plan your next few steps. If the environment outputs 'Nothing happened', that means the previous action is invalid and you should try more options. 
Reminder: the action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal. 

There are four thinking levels:
Level 1 - Instinctive Response: Immediate reaction based on intuition, no analysis.
Level 2 - Situational Awareness: Assess current state and available actions before acting.
Level 3 - Experience Integration: Reflect on past actions and outcomes to inform current decisions.
Level 4 - Strategic Planning: Assess the task goal, past lessons, and current state to analyze the future impact of each candidate action and optimize the decision.

At each step, you must first choose an appropriate level of thinking (one of the four levels) to respond based on the given scenario. The chosen level MUST be enclosed within <level> </level> tags.
Next, reason through the problem step-by-step using the chosen thinking level. This reasoning process MUST be enclosed within <think> </think> tags. For Level 1 (Instinctive Response), use the fixed text: "Okay, I think I have finished thinking." For Levels 2-4, provide detailed reasoning as shown in examples.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.

[Output Format] 
Your output must adhere to the following format:
EXAMPLE 1:
<level>1</level>
<think>Okay, I think I have finished thinking.</think>
<action>your_next_action</action>

EXAMPLE 2:
<level>2</level>
<think>
Current state: [Current state and inventory]
Available actions: [What actions are valid right now]
Reasoning: [Choose the best action and explain why]
</think>
<action>your_next_action</action>

EXAMPLE 3:
<level>3</level>
<think>
Goal: [What needs to be accomplished]
Current state: [Current state and inventory]
Available actions: [What actions are valid right now]
Reflection: [How effective were recent actions, what was learned]
Reasoning: [Choose the best action based on experience]
</think>
<action>your_next_action</action>

EXAMPLE 4:
<level>4</level>
<think>
Goal: [What needs to be accomplished]
Current state: [Current state and inventory]
Available actions: [What actions are valid right now]
Reflection: [How effective were recent actions, what was learned]
Evaluation: [Assess the potential effectiveness of each candidate action]
Reasoning: [Choose the optimal action with strategic reasoning]
</think>
<action>your_next_action</action>

{task_description}
Your current observation is: {current_observation}

Now it's your turn to generate next step response.
""".strip()


ALFWORLD_TEMPLATE_ADA = """
Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. For each of your turn, you will be given a list of actions which you can choose one to perform in this turn. After each turn, the environment will give you immediate feedback based on which you plan your next few steps. If the environment outputs 'Nothing happened', that means the previous action is invalid and you should try more options. 
Reminder: the action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal. 

There are four thinking levels:
Level 1 - Instinctive Response: Immediate reaction based on intuition, no analysis.
Level 2 - Situational Awareness: Assess current state and available actions before acting.
Level 3 - Experience Integration: Reflect on past actions and outcomes to inform current decisions.
Level 4 - Strategic Planning: Assess the task goal, past lessons, and current state to analyze the future impact of each candidate action and optimize the decision.

At each step, you must first choose an appropriate level of thinking (one of the four levels) to respond based on the given scenario. The chosen level MUST be enclosed within <level> </level> tags.
Next, reason through the problem step-by-step using the chosen thinking level. This reasoning process MUST be enclosed within <think> </think> tags. For Level 1 (Instinctive Response), use the fixed text: "Okay, I think I have finished thinking." For Levels 2-4, provide detailed reasoning as shown in examples.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.

[Output Format] 
Your output must adhere to the following format:
EXAMPLE 1:
<level>1</level>
<think>Okay, I think I have finished thinking.</think>
<action>your_next_action</action>

EXAMPLE 2:
<level>2</level>
<think>
Current state: [Current state and inventory]
Available actions: [What actions are valid right now]
Reasoning: [Choose the best action and explain why]
</think>
<action>your_next_action</action>

EXAMPLE 3:
<level>3</level>
<think>
Goal: [What needs to be accomplished]
Current state: [Current state and inventory]
Available actions: [What actions are valid right now]
Reflection: [How effective were recent actions, what was learned]
Reasoning: [Choose the best action based on experience]
</think>
<action>your_next_action</action>

EXAMPLE 4:
<level>4</level>
<think>
Goal: [What needs to be accomplished]
Current state: [Current state and inventory]
Available actions: [What actions are valid right now]
Reflection: [How effective were recent actions, what was learned]
Evaluation: [Assess the potential effectiveness of each candidate action]
Reasoning: [Choose the optimal action with strategic reasoning]
</think>
<action>your_next_action</action>

{task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: 
{action_history}
You are now at step {current_step} and your current observation is: {current_observation} 

Now it's your turn to generate next step response. 
""".strip()
