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
import os
from typing import List

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.generativeai import caching
from loguru import logger
from flask import Flask, request, jsonify
from flask_cors import CORS
import datetime

# Define our request defaults
model = "gemini-1.5-pro-001"
chunk_size=500000
temperature = 0.1
max_tokens = 8192
top_p = 1.0
frequency_penalty = 0.0
presence_penalty = 0.0
log_level = "INFO"

# Base method to get a genai_model object to use with either sending requests or getting accurate token counts
# Does not leverage the context cache is used.
def get_genai_model():
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
    
    generation_config = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": 40,
        "max_output_tokens": max_tokens,
    }

    # Define our non cache
    genai_model = genai.GenerativeModel(model_name=model, generation_config=generation_config, safety_settings=safety_settings)
    return genai_model

# Validates that the required environment variables are set.
def check_required_env_vars():
    """Check required environment variables"""
    required_env_vars = [
        "GEMINI_API_KEY",
    ]
    for required_env_var in required_env_vars:
        if os.getenv(required_env_var) is None:
            raise ValueError(f"{required_env_var} is not set")

# Defines the based prompt used to build the context cache.
def context_prompt(context: str):
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
    """
    return prompt 

# Defines the prompt used for a specific question.  Designed to use used in tandem with the context cache.
def question_prompt(message: str):
    prompt = f"""
    QUESTION:
    {message}
    END QUESTION
    
    Please begin your response now:
    """
    return prompt 

# Defines a prompt to be used when a code base is not supplied or the code base would be too large.  This lacks the code base 
# context entirely and probably doesn't end up being super useful in these instances outside of a general GPT with Gemini
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


def build_cached_content(
        context:str,
        cache_key: str,
        model: str):
    genai_model = get_genai_model()
    token_count = 0

    prompt_for_context = context_prompt(context)
    token_count = genai_model.count_tokens(prompt_for_context)
    logger.debug(f"Context token count: {token_count}")

    # Set an arbitrary limit (2,000,000 token limit) to avoid the final question and prompt from being too large.
    # This can probably be larger in reality.
    if token_count.total_tokens > 1900000:
        return ["ERROR: Context too large", ""]
        
    # A cache can only be built if the min tokens is 32768, else it's too small to bother.  Skip the step but report OK
    if (token_count.total_tokens < 32768):
        return ["OK", ""]
   
    cache = ""
    exists = False

    for c in caching.CachedContent.list():
        logger.debug(f"Looking at cache: {c.display_name}")
        if c.display_name == cache_key:
            exists = True
            cache = c

    if not exists:           
        # Use the cache_key and upload the project as a cache
        logger.debug(f"Caching with cache key {cache_key} - {model}") 
        cache = caching.CachedContent.create(
            model= model,
            contents=prompt_for_context, 
            display_name=cache_key, 
            ttl=datetime.timedelta(minutes=5)
        ) 
        logger.debug("Finished creating context cache")

    return ["OK", cache]

# Base method to take input and send a question to Gemini
def answer_question(
        context: str,
        message: str,
        cache_key: str,
        model: str
):
    
    # Define our non cache model
    genai_model = get_genai_model()
   
   # Check out context to see if it's too big
    prompt_for_question = question_prompt(message)

    result = build_cached_content(context=context, cache_key=cache_key, model=model)
    if result[0] != "OK":
        return f"Error building context cache: {result[0]}"
    
    cache = result[1]

    if cache != "":
        logger.debug("Using context cache to init model")
        genai_model = genai.GenerativeModel.from_cached_content(cached_content=cache)
        logger.debug("Genearting content respond")
        response = genai_model.generate_content([(
            prompt_for_question
        )])
        logger.debug("Done genearating content response")
        conversation_result = clean_response(response.text)
        return conversation_result
    else:
        # If we're here we did not get an error and our token count is probably too low to cache.
        # Build a prompt from message_prompt and 
        prompt_for_context = context_prompt(context)
        response = genai_model.generate_content([(
            f"{prompt_for_context}\n\n{prompt_for_question}"
        )])
        conversation_result = clean_response(response.result)
        return conversation_result

# Iterates over the project files and returns an array of object designed to be used directly with Gemini [0] and a listing of 
# directory paths to be used for display [1]
def read_project_files(exclude_dirs=['.github', '.git', '.cm', '.idea', 'webpack', 'spec', 'script', 'benchmarks', 'bin', 'benchmarks', 'log', 'node_modules', 'dist', 'fixtures']):
    project_content = []
    dir_paths = []
    for root, dirs, files in os.walk('code'):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        dir_paths.append(f"{root}\n")
        for file in files:
            if file.endswith(('.py', '.json', '.kt', '.html', '.js', '.cs', '.qml', '.asp', '.vb',
                              '.ts', '.java', '.c', '.cpp', '.h', '.hpp', '.go', '.rs', '.swift',
                              '.sh', '.rb', '.php', '.mdx', '.rs', '.sql')):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    content = f.read()
                project_content.append(f"File: {file_path}\n\n ```{content}```\n\n")
    return ['\n'.join(project_content), dir_paths]

# Cleans the response for display as markdown in the browser.  Trial and error finding markdown bits that broke styling on the web.
def clean_response(answer):
    response = answer
    response = answer.replace("\n", "  \n")
    response = answer.replace("```", "``` ")
    response = response.strip('"')
    return response


def send_message(
        message: str,
        include_code: bool,
        cache_key: str,
        model: str,
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

    project_content = [""]
    if include_code:
        # Read the entire project content
        logger.debug("Reading project files")
        project_content = read_project_files()
        project_size = len(project_content[0])
        logger.debug(f"Project Size: {project_size}")
    
    # Ask the question
    answer = answer_question(
        context=project_content[0],
        message=message,
        cache_key=cache_key,
        model=model
    )
    return answer
    
# Define a web server to receive incoming requests to interact with Gemini
app = Flask(__name__, static_folder='chat_bot', static_url_path='')
CORS(app)

@app.route('/query/', methods=['POST'])
def query_route():
  """Receives POST request with JSON data and returns a response."""
  try:

    # Extract data from incoming request.
    data = request.get_json()
    message = data.get('message')
    include_code = data.get('include_code')
    cache_key = data.get("cache_key")

    if not message:
      return jsonify({'error': 'Missing "message" in JSON body'}), 400

    logger.debug("Calling send_message")
    # Send our message to the model for a response
    answer = send_message(message=message, 
                          include_code = include_code,
                          cache_key = cache_key,
                          model=model,
                          log_level=log_level)
    return jsonify({'answer': f'{answer}'}), 200
  except Exception as e:
    logger.debug(f"Exception occurred: {e}")
    return jsonify({'error': str(e)}), 500

@app.route('/context_stats/', methods=['POST'])
def context_stats_route():
    """
        Counts tokens of the project files using the specified model and returns that is read that builds that context.
    """
    genai_model = get_genai_model()
    # Extract included dirs (long term)
    # Read project files appropriately and build out context for files
    project_content = read_project_files()
    # Count tokens
    token_count = genai_model.count_tokens(project_content[0])
    # Return JSON with token count
    return jsonify({'file_paths': project_content[1], 'token_count': token_count.total_tokens}), 200



if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)