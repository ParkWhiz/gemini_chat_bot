# Instructions

## Pre-Requisites
Docker and Chrome/Firefox.  Edge? Maybe, idk.

## Get Gemini API Key
You'll need this as an arg to start the service

## Run via Docker
For convenience the script `run_service.sh` can be leveraged.
```./run_service.sh GEMINI_API_KEY YOUR_CODE_DIRECTORY```

## Open Chat Bot and submit a request.
Navigate to `http://127.0.0.1:5000/index.html`

Enter your query, hit enter/submit, be patient, especially on a large code base.   The first upload will generate a context cache with your code base context so subsequent queries will be speedier.

Note: If you are on MacOS Monterey or later and have AirPlay enabled, then you will be unable to use port 5000. You must either disable AirPlay or run this project on a different port.
