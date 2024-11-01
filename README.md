# Instructions

## Pre-Requisites
Docker and Chrome/Firefox.  Edge? Maybe, idk.

## Pull the latest docker image
Currently `parkwhiz/chat_bot:0.1`

## Get Gemini API Key
You'll need this as an arg to start the service

## Run via Docker
For convenience the script `run_service.sh` can be leveraged.
```./run_service.sh GEMINI_API_KEY YOUR_CODE_DIRECTORY```

## Open Chat Bot and submit a request.
Navigate to `http://127.0.0.1:5000/index.html`

Enter your query, hit submit, be patient, especially on a large code base.