docker pull parkwhiz/chat_bot:0.1
docker rm chat_bot
docker run -v /Users/jkrohn/Documents/code/parkwhiz-js:/app/code -p 5000:5000 --name chat_bot -e GEMINI_API_KEY="ADD_KEY_HERE" -i parkwhiz/chat_bot:0.1