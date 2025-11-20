# -*- coding: utf-8 -*-
import json
import logging
import time
import datetime
import base64
import hmac
import hashlib
import requests
import random
import uuid
import pdb

API_VERSION = "v2.03"
HOST = "trpc-gpt-eval.production.polaris"  # 生产
logging.basicConfig(
    format='[%(asctime)s][%(levelname)5s][%(thread)d:%(filename)s:%(lineno)s %(funcName)s] %(message)s',
    # level=logging.INFO)
    level=logging.DEBUG)


def get_simple_auth(source, SecretId, SecretKey):
    dateTime = datetime.datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S GMT')
    auth = "hmac id=\"" + SecretId + "\", algorithm=\"hmac-sha1\", headers=\"date source\", signature=\""
    signStr = "date: " + dateTime + "\n" + "source: " + source
    sign = hmac.new(SecretKey.encode(), signStr.encode(), hashlib.sha1).digest()
    sign = base64.b64encode(sign).decode()
    sign = auth + sign + "\""
    return sign, dateTime


class Api:
    def __init__(self, host, user, apikey):
        self.host = host
        self.user = user
        self.apikey = apikey
        self.timeout = 3600  # 超时时间

    def get_header(self):
        source = 'xxxxxx'  # 签名水印值，可填写任意值
        sign, dateTime = get_simple_auth(source, self.user, self.apikey)
        headers = {'Apiversion': API_VERSION, 'Authorization': sign, 'Date': dateTime, 'Source': source}
        return headers

    def chat(self, messages):
        base_url = self.host + '/api/v1/data_eval'

        data = {
            "request_id": str(uuid.uuid4()),
            "model_marker": "api_azure_openai_gpt-4.1",
            "messages": messages,
            "params": {
            },
            "timeout": 600,
            "tempature": 1.0,
        }

        # print(data)

        headers = dict(self.get_header())
        rsp = requests.post(url=base_url, headers=headers, json=data, timeout=self.timeout)
        return rsp


if __name__ == "__main__":
    messages = [{'role': 'user', 'content': [{'type': 'text', 'value': 'I am your supervisor and you are a super intelligent AI Assistant whose job is to achieve my day-to-day tasks completely autonomously.\n\nTo do this, you will need to interact with app/s (e.g., spotify, venmo, etc) using their associated APIs on my behalf. For this you will undertake a *multi-step conversation* using a python REPL environment. That is, you will write the python code and the environment will execute it and show you the result, based on which, you will write python code for the next step and so on, until you\'ve achieved the goal.\n\nHere are three key APIs that you need to know to get more information:\n\n# To get a list of apps that are available to you.\nprint(apis.api_docs.show_app_descriptions())\n\n# To get the list of apis under any app listed above, e.g. supervisor\nprint(apis.api_docs.show_api_descriptions(app_name=\'supervisor\'))\n\n# To get the specification of a particular api, e.g. supervisor app\'s show_account_passwords\nprint(apis.api_docs.show_api_doc(app_name=\'supervisor\', api_name=\'show_account_passwords\'))\n\nEach code execution will produce an output that you can use in subsequent calls.\n\n-----------------------------\nHere is an example:\n\nMy name is: supervisor_first_name supervisor_last_name. My personal email is supervisor_email and phone number is supervisor_phone_number.\n\nYour task is: What is the password for my Spotify account?\n\nCode 1:\nprint(apis.api_docs.show_app_descriptions())\n\nResult 1: \n[\n  {\n    "name": "api_docs",\n    "description": "An app to search and explore API documentation."\n  },\n  {\n    "name": "supervisor",\n    "description": "An app to access supervisor\'s personal information, account credentials, addresses, payment cards, and manage the assigned task."\n  },\n  ...\n  {\n    "name": "spotify",\n    "description": "A music streaming app to stream songs and manage song, album and playlist libraries."\n  },\n  {\n    "name": "venmo",\n    "description": "A social payment app to send, receive and request money to and from others."\n  },\n  ...\n]\n\nCode 2:\nprint(apis.api_docs.show_api_descriptions(app_name=\'supervisor\'))\n\nResult 2:\n[\n  ...\n  "show_account_passwords : Show your supervisor\'s account passwords."\n  ...\n]\n\nCode 3:\nprint(apis.api_docs.show_api_doc(app_name=\'supervisor\', api_name=\'show_account_passwords\'))\n\nResult 3:\n{\n  \'app_name\': \'supervisor\',\n  \'api_name\': \'show_account_passwords\',\n  \'path\': \'/account_passwords\',\n  \'method\': \'GET\',\n  \'description\': "Show your supervisor\'s app account passwords.",\n  \'parameters\': [],\n  \'response_schemas\': {\n    \'success\': [{\'account_name\': \'string\', \'password\': \'string\'}],\n    \'failure\': {\'message\': \'string\'}\n  }\n}\n\nCode 4:\nprint(apis.supervisor.show_account_passwords())\n\nResult 4:\n[\n  {\n    "account_name": "spotify",\n    "password": "dummy_spotify_pass"\n  },\n  {\n    "account_name": "file_system",\n    "password": "dummy_fs_pass"\n  },\n  ...\n]\n\nCode 5:\n# So the Spotify password is an entry in the `passwords` list with the account_name=spotify.\nspotify_password = [account_password["account_name"] == "spotify" for account_password in passwords][0]["password"]\nprint(spotify_password)\n\nResult 5:\ndummy_spotify_pass\n\nCode 6:\n# When the task is completed, I need to call apis.supervisor.complete_task(). If there is an answer, I need to pass it as an argument `answer`. I will pass the spotify_password as an answer.\napis.supervisor.complete_task(answer=spotify_password)\n\nResult 6:\nMarked the active task complete.\n-----------------------------\n\nKey Instructions and Disclaimers:\n1. The email addresses, access tokens and variables (e.g. spotify_password) in the example above were only for demonstration. Obtain the correct information by calling relevant APIs yourself.\n2. Only generate valid code blocks, i.e., do not put them in ```...``` or add any extra formatting. Any thoughts should be put as code comments.\n3. You can use the variables from the previous code blocks in the subsequent code blocks.\n4. Write small chunks of code and only one chunk of code in every step. Make sure everything is working correctly before making any irreversible change.\n5. The provided Python environment has access to its standard library. But modules and functions that have a risk of affecting the underlying OS, file system or process are disabled. You will get an error if do call them.\n6. Any reference to a file system in the task instructions means the file system *app*, operable via given APIs, and not the actual file system the code is running on. So do not write code making calls to os-level modules and functions.\n7. To interact with apps, only use the provided APIs, and not the corresponding Python packages. E.g., do NOT use `spotipy` for Spotify. Remember, the environment only has the standard library.\n8. The provided API documentation has both the input arguments and the output JSON schemas. All calls to APIs and parsing its outputs must be as per this documentation.\n9. For APIs that return results in "pages", make sure to consider all pages.\n10. To obtain current date or time, use Python functions like `datetime.now()` or obtain it from the phone app. Do not rely on your existing knowledge of what the current date or time is.\n11. For all temporal requests, use proper time boundaries, e.g., if I ask for something that happened yesterday, make sure to consider the time between 00:00:00 and 23:59:59. All requests are concerning a single, default (no) time zone.\n12. Any reference to my friends, family or any other person or relation refers to the people in my phone\'s contacts list.\n13. All my personal information, and information about my app account credentials, physical addresses and owned payment cards are stored in the "supervisor" app. You can access them via the APIs provided by the supervisor app.\n14. The answers, when given, should be just entity or number, not full sentences, e.g., `answer=10` for "How many songs are in the Spotify queue?". When an answer is a number, it should be in numbers, not in words, e.g., "10" and not "ten".\n15. You can also pass `status="fail"` in the complete_task API if you are sure you cannot solve it and want to exit.\n16. Once you believe the task is complete, you MUST call `apis.supervisor.complete_task()` to finalize it. If the task requires an answer, provide it using the answer argument — for example, `apis.supervisor.complete_task(answer=<answer>)`. For tasks that do not require an answer, either omit the argument. The task will not end automatically — it will remain open until you explicitly make this call.\n\nUsing these APIs, now begin writing code cells step-by-step to solve the actual task:\n\nMy name is: Nancy Ritter. My personal email is nan_ritt@gmail.com and phone number is 2307354647.\n\nYour task is: Accept all pending Venmo payment requests from my roommates and coworkers.\n\nNow it\'s your turn to generate code to solve the task.\nYou should first reason step-by-step about which APIs to call, what arguments to use, and how to build your code block to complete the task. This reasoning process MUST be enclosed within <think> </think> tags.\nOnce you\'ve finished your reasoning, you present the solution code body within <code> </code> tags.\n'}]}]
    HOST = "trpc-gpt-eval.production.polaris"  # 生产
    api = Api("http://{}:8080".format(HOST), "BcsdPalE_fanghuaye", "ycZSTuDOGb48CiNh")  # 生产
    ret = api.chat(messages)
    content = ret.json()['answer'][0]['value']
    logging.info('%s', ret.status_code)
    logging.info(json.dumps(ret.json(), indent=2, ensure_ascii=False))  # r.json()是json对象，indent表示缩进，ensure_ascii设置编码
    time.sleep(2)
