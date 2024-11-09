set -euxo pipefail
TP_SIZE=${1:-2}  # Use first argument if provided, otherwise default to 2
HF_HOME=/models/hub/

python3 mistral-tensorrt-llm-server/download.py --output_dir /models/mistral-nemo-huggingface
git clone https://github.com/NVIDIA/TensorRT-LLM.git || true
cd TensorRT-LLM/examples/llama/
git config --global --add safe.directory /workspace/TensorRT-LLM
git checkout b7868dd1bd1186840e3755b97ea3d3a73ddd76c5
python3 convert_checkpoint.py \
   --model_dir /models/mistral-nemo-huggingface/ \
   --output_dir /models/mistral-instruct/converted/ \
   --tp_size $TP_SIZE --dtype bfloat16

trtllm-build --checkpoint_dir /models/mistral-instruct/converted/ \
            --output_dir /models/mistral-instruct/tensorrt_format/ \
            --max_input_len 2048 \
            --max_seq_len 4096
