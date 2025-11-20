# Load model directly
import pdb
from openai import OpenAI
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
max_gpu_memory = 90
start_id = 0
num_gpus = 4

tokenizer = AutoTokenizer.from_pretrained("/apdcephfs_cq11_1567347/share_1567347/share_info/llm_models/models--openai--gpt-oss-120b/snapshots/8b193b0ef83bd41b40eb71fee8f1432315e02a3e")
# model = AutoModelForCausalLM.from_pretrained("/apdcephfs_cq11_1567347/share_1567347/share_info/llm_models/models--openai--gpt-oss-120b/snapshots/8b193b0ef83bd41b40eb71fee8f1432315e02a3e",
#                                             device_map="auto",
#                                            max_memory={i: f"{max_gpu_memory}GiB" for i in range(start_id, start_id + num_gpus)},
# )

from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    Role,
    SystemContent,
    ToolNamespaceConfig,
    ToolDescription,
    load_harmony_encoding,
    ReasoningEffort
)


encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
 
task = "Retrieve the English Wikipedia page for Mercedes Sosa (as of the latest 2022 version). Extract the list of studio albums and their release years. Count how many studio albums were published between 2000 and 2009 inclusive, and return that number. Provide a brief citation of sources used (e.g., URLs and sections)."

tool_desc = [
                ToolDescription.new(
                    "search",
                    "Perform web searches using Google Serper API",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "array",
                                "items": {'type': 'string'},
                                "description": "Array of search query strings. Can search multiple queries in one call.",
                            },
                            "num_results": {
                                "type": "integer",
                                "description": "Number of results to return per query (default: 5)",
                                "default": 5
                            },
                        },
                        "required": ['query'],
                    },
                ),
                ToolDescription.new(
                    "visit",
                    "Visit web pages and extract relevant content based on goals",
                    parameters={
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "array",
                                "items": {'type': 'string'},
                                "description": "Array of URLs to visit. Can be a single URL or multiple URLs.",
                            },
                            "goal": {
                                "type": "string",
                                "description": "The specific information goal for visiting the webpage(s)"
                            },
                        },
                        "required": ['url', 'goal'],
                    },
                ),
                ToolDescription.new(
                    "code_execution",
                    "Execute Python code safely and return results",
                    parameters={
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code to execute",
                            },
                            "language": {
                                "type": "string",
                                "description": "Programming language (currently only 'python' is supported)",
                                "default": "python"
                            },
                        },
                        "required": ['code'],
                    },
                ),
            ]

system_message = (
    SystemContent.new()
        .with_reasoning_effort(ReasoningEffort.LOW)
        .with_conversation_start_date("2025-06-28")
        .with_knowledge_cutoff("2024-06")
        .with_required_channels(["analysis", "commentary", "final"])
        .with_tools(ToolNamespaceConfig(name="default", tools=tool_desc))
)
 
developer_message = (
    DeveloperContent.new()
    .with_instructions("You are a specialized Researcher Agent agent in a multi-agent system.\n\n## Guidelines\n- Focus solely on your assigned task\n- Be thorough, step-by-step, and methodical\n- Use tools systematically to gather information or perform actions\n- Think before taking any action")
)

messages = [
        Message.from_role_and_content(Role.SYSTEM, system_message),
        Message.from_role_and_content(Role.DEVELOPER, developer_message),
        Message.from_role_and_content(Role.USER, task),
        Message.from_role_and_content(
            Role.ASSISTANT,
            'We need Wikipedia page as of latest 2022 version. Use history? We can retrieve page revision from end of 2022. Use search.',
        ).with_channel("analysis"),
        Message.from_role_and_content(Role.ASSISTANT, '{\n  "path": "",\n  "query": "Mercedes Sola Wikipedia 2022 revision",\n  "max_results": 10\n}')
        .with_channel("commentary")
        .with_recipient("default.search"),
        Message.from_author_and_content(
            Author.new(Role.TOOL, "default.search"),
            "<stdout>A Google search for 'Mercedes Sola Wikipedia 2022 revision' found 5 results:\n\n## Web Results\n1. [Mercedes Sola - Wikipedia](https://en.wikipedia.org/wiki/Mercedes_Sola)\nMissing: 2022 revision\n\n2. [Mercedes Sosa - Wikipedia](https://en.wikipedia.org/wiki/Mercedes_Sosa)\nHaydée Mercedes Sosa (9 July 1935 – 4 October 2009) was an Argentine singer who was popular throughout Latin America and many countries outside the region.\n\n3. [Mercedes Sosa - Wikipedia, a enciclopedia libre](https://gl.wikipedia.org/wiki/Mercedes_Sosa)\nHaydee Mercedes Sosa, nada en San Miguel de Tucumán o 9 de xullo de 1935 e finada o 4 de outubro de 2009 en Buenos Aires, foi unha cantante de raíz ...\n\n4. [Dean's List - Fashion Institute of Technology](https://www.fitnyc.edu/academics/academic-divisions/business-and-technology/about/current/deans-list.php)\nThe Dean's List, posted at the end of each semester, honors those students who have completed 12 or more credits (may not include courses taken on a pass/fail ...\n\n5.</stdout>\n<stderr></stderr>\n<returncode>0</returncode>",
        ).with_recipient("assistant").with_channel("commentary"),
    ]
convo = Conversation.from_messages(messages)

tokens = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)

# inputs = {
#    "input_ids": torch.tensor([tokens]).to(model.device),  
#     "attention_mask": torch.ones(1, len(tokens)).to(model.device) 
# } 
   
# outputs = model.generate(**inputs, 
#                          max_new_tokens=8192,
#                          temperature=0.3)

# parsed = encoding.parse_messages_from_completion_tokens(outputs[0][inputs["input_ids"].shape[-1]:], Role.ASSISTANT)
# print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]).strip())
# print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]).strip())
pdb.set_trace()
prompt = [{'role': 'user', 'content': tokenizer.decode(tokens)}]
client = OpenAI(base_url="http://29.81.228.199:8034/v1",
                    api_key="EMPTY")
completion = client.chat.completions.create(
                model="/apdcephfs_cq11_1567347/share_1567347/share_info/llm_models/models--openai--gpt-oss-120b/snapshots/8b193b0ef83bd41b40eb71fee8f1432315e02a3e",
                messages=prompt,
                temperature=0.3,
                max_tokens=8192,
                )
print(completion.choices[0].message.content.strip())

