import argparse
from src.model import LlamaModel


def main():
    print("Hello from bias-a-thon!")

    parser = argparse.ArgumentParser(
        prog="Llama Inference", description="2025-Bias-A-Thon"
    )
    parser.add_argument("--title", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    llama = LlamaModel(
        title=args.title, prompt_file_name=args.prompt, is_test=args.test
    )

    print("Load llama model")


if __name__ == "__main__":
    main()
