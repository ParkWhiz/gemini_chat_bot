API_KEY=$1
if [ -e $API_KEY ]
    then
        echo "You must provide an API Key for Gemini to continue"
        exit 1
fi

CODE_PATH=$2
if [ ! -d "$CODE_PATH" ]
    then
        echo "You must provide a working code path (directory)"
        exit 1
fi

echo "Stopping existing container"
cur_proc=$(docker ps | grep chat_bot)
if [[ ! -e $proc ]]; then docker rm -f chat_bot; fi

echo "Building image"
docker build -t chat_bot .

echo "Chat bot can be accessed at http://127.0.0.1:5000/index.html"
docker run -v $CODE_PATH:/app/code -p 5000:5000 --name chat_bot -e GEMINI_API_KEY="$API_KEY" -i chat_bot
