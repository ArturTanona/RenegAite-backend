set -euxo pipefail
TP_SIZE=${1:-2}  # Use first argument if provided, otherwise default to 2


python3 download.py --output_dir /models/mistral-nemo-huggingface
python3 convert_checkpoint.py \
   --model_dir /models/mistral-nemo-huggingface/ \
   --output_dir /models/mistral-instruct/converted/ \
   --tp_size $TP_SIZE --dtype bfloat16

trtllm-build --checkpoint_dir /models/mistral-instruct/converted/ \
            --output_dir /models/mistral-instruct/tensorrt_format/ \
            --max_input_len 2048 \
            --max_seq_len 4096