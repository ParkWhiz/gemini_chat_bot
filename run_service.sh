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

docker pull parkwhiz/chat_bot:0.1
docker rm chat_bot
docker run -v $CODE_PATH:/app/code -p 5000:5000 --name chat_bot -e GEMINI_API_KEY="$API_KEY" -i parkwhiz/chat_bot:0.1