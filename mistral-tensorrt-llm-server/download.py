# noqa: D100
import argparse
import os

from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        default="/models/mistral-nemo-huggingface/",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Mistral-Nemo-Instruct-2407",
        help="Name or path of the model to download from HuggingFace",
    )
    args = parser.parse_args()
    breakpoint()
    os.makedirs(args.output_dir, exist_ok=True)

    token = os.environ.get("HF_TOKEN")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, token=token)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=token)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
