# RenegAite-backend
This is tensorrtllm backend for RenegAite - an open app for navigating through AI Act. 

The is rather experimental setup and demo.  Not recommend to use it in production.

# Instructions

Set up .env file in the root of the project with the following:

```
OPENAI_API_KEY=
HF_TOKEN=
BACKEND_URL=http://mistral-tensorrt-llm-server:8000 # name of the service in docker-compose.yml - do not change
```

Run the following command to start the backend:

```
docker compose build
docker compose run tensorrtllm-worker mistral-tensorrt-llm-server/prepare.sh
docker compose up
```



# Host own endpoint:

1. Set up your own free account on ngrok and go to https://dashboard.ngrok.com/get-started/setup/macos or https://ngrok.com/docs/getting-started/?os=macos (other platforms are also supported)
2. Run `ngrok http 8002 --host-header="localhost:8002"` # you can change the port in the docker-compose.yml file. Use mechanics of your OS to run ngrok in the background and be aware of its auth key
3. Add it in the "Custom Endpoint" section in https://renegaite.com/  

# How was TensorRT-LLM used?
It was used to build a server that fulfills V1 OpenAI API. 

# How was used LLamaIndex?

It was used to build a tool that allows to search the documents using natural language. It gets the input from the renegaite.com chat and sends it to the LLM. Then it gets the response and sends it back to the chat. 

# License

Part of the code was taken from the [tensorrt-llm](https://github.com/NVIDIA/TensorRT-LLM) repository.
