#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import json
import os
from typing import List

import click
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import requests
from loguru import logger
from flask import Flask, request, jsonify
from flask_cors import CORS
import tiktoken

def count_tokens(text, model_name):
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    return len(tokens)

def check_required_env_vars():
    """Check required environment variables"""
    required_env_vars = [
        "GEMINI_API_KEY",
    ]
    for required_env_var in required_env_vars:
        if os.getenv(required_env_var) is None:
            raise ValueError(f"{required_env_var} is not set")


def chunk_string(input_string: str, chunk_size) -> List[str]:
    """Chunk a string"""
    chunked_inputs = []
    for i in range(0, len(input_string), chunk_size):
        chunked_inputs.append(input_string[i:i + chunk_size])
    return chunked_inputs

def full_prompt(message: str, context: str):
    prompt = f"""
    You are answering a question for a software developer. The context provided includes the entire project structure and content, followed by a question.

    As an excellent software engineer analyze the project and answer the question to the best of your ability.

    Provide a response that includes:
    - Guidance on where in the code base to look to answer the question.
    - Responses should include relevant file names, method names, data strutures or any other relevant information.
    - A explanation of why this appears to be the relevant location to answer the question
    - Format the response in Markdown.

    DO NOT PROVIDE ANY RESPONSE UNTIL YOU SEE THE PROMPT "Please begin your response now:"

    CONTEXT:
    {context}
    END CONTEXT

    QUESTION:
    {message}
    END QUESTION
    
    Please begin your response now:
    """
    return prompt

def partial_prompt(message: str):
    prompt = f"""
    You are answering a question for a software developer.

    As an excellent software engineer analyze the question, any approrpriate reference material, and answer the question to the best of your ability.

    Provide a response that includes:
    - Guidance on best principles and practices
    - Code examples where relevant and cite sources where possible.
    - A explanation of why the response answers the question or solves the problem
    - Format the response in Markdown.

    DO NOT PROVIDE ANY RESPONSE UNTIL YOU SEE THE PROMPT "Please begin your response now:"

    QUESTION:
    {message}
    END QUESTION
    
    Please begin your response now:
    """
    return prompt

def answer_question(
        context: str,
        message: str,
        include_code: bool,
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        frequency_penalty: float,
        presence_penalty: float,
        prompt_chunk_size: int
):
    """Answer a question about a code base"""
    generation_config = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": 40,
        "max_output_tokens": max_tokens,
    }
    
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
    
    genai_model = genai.GenerativeModel(model_name=model, generation_config=generation_config, safety_settings=safety_settings)
    # Check out context to see if it's too big
    token_count = 0
    if include_code:
        token_count = genai_model.count_tokens(context)
        if token_count.total_tokens > 1900000:
            context = "TOO LARGE, OMITTED"
        logger.debug(f"Code Base token count: {token_count}")

    # Combine context and diff
    prompt = ""
    if include_code:
        prompt = full_prompt(message=message, context=context)
    else:
        prompt = partial_prompt(message=message)

    full_token_count = genai_model.count_tokens(prompt)
    logger.debug(f"Full token count: {full_token_count}")

    # Chunk the input if necessary
    chunked_inputs = chunk_string(input_string=prompt, chunk_size=prompt_chunk_size)
    
    convo = genai_model.start_chat(history=[
        {
            "role": "user",
            "parts": [prompt]
        }
    ])

    for chunked_input in chunked_inputs:
        logger.debug(f"Chunked input...")
        convo.send_message(chunked_input)
        logger.debug(f"Message Sent...")



    logger.debug("Done sending, cleaning repsonse.")
    conversation_result = clean_response(convo.last.text)
    #logger.debug(f"Response AI: {conversation_result}")

    return conversation_result


def read_project_files(exclude_dirs=['.github', '.git', '.cm', '.idea', 'webpack', 'spec', 'script', 'benchmarks', 'bin', 'benchmarks', 'log']):
    project_content = []
    for root, dirs, files in os.walk('code'):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            if file.endswith(('.py', '.json', '.kt', '.html', '.js', '.cs', '.qml', '.asp', '.vb',
                              '.ts', '.java', '.c', '.cpp', '.h', '.hpp', '.go', '.rs', '.swift',
                              '.sh', '.rb', '.php', '.mdx', '.rs', '.sql')):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    content = f.read()
                project_content.append(f"File: {file_path}\n\n ```{content}```\n\n")
    return '\n'.join(project_content)

def clean_response(answer):
    response = answer
    response = answer.replace("\n", "  \n")
    response = answer.replace("```", "``` ")
    response = response.strip('"')
    return response

def send_message(
        message: str,
        include_code: bool,
        chunk_size: int,
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        frequency_penalty: float,
        presence_penalty: float,
        log_level: str
):
    # Set log level
    logger.level(log_level)
    # Check if necessary environment variables are set or not
    check_required_env_vars()

    # Set the Gemini API key
    api_key = os.getenv("GEMINI_API_KEY")
    # Use Rest as the transport agent
    genai.configure(api_key=api_key, transport="rest")

    project_content = ""
    if include_code:
        # Read the entire project content
        project_content = read_project_files()
        project_size = len(project_content)
        logger.debug(f"Project Size: {project_size}")

    #logger.debug(f"Project content: {project_content}")
    
    # Request a code review
    chunked_answer = answer_question(
        context=project_content,
        message=message,
        include_code=include_code,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        prompt_chunk_size=chunk_size
    )
    #debug(f"Chunked reviews: {chunked_answer}")
    return chunked_answer
    
# Define a web server to receive incoming requests to interact with Gemini

app = Flask(__name__, static_folder='chat_bot', static_url_path='')
CORS(app)

@app.route('/query/', methods=['POST'])

def receive_data():
  """Receives POST request with JSON data and returns a response."""
  try:

    # Define our request defaults
    model = "gemini-1.5-pro"
    chunk_size=500000
    temperature = 0.1
    max_tokens = 8192
    top_p = 1.0
    frequency_penalty = 0.0
    presence_penalty = 0.0
    log_level = "INFO"

    # Extract data from incoming request.
    data = request.get_json()
    message = data.get('message')
    include_code = data.get('include_code')

    if not message:
      return jsonify({'error': 'Missing "message" in JSON body'}), 400

    # Send our message to the model for a response
    answer = send_message(message=message, 
                          include_code = include_code,
                          chunk_size=chunk_size,
                          model=model,
                          temperature=temperature,
                          max_tokens=max_tokens,
                          top_p=top_p,
                          frequency_penalty=frequency_penalty,
                          presence_penalty=presence_penalty,
                          log_level=log_level)

    return jsonify({'answer': f'{answer}'}), 200
  except Exception as e:
    logger.debug(f"Exception occurred: {e}")
    return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)