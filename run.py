"""Project entry point; heavy libraries are imported only for the chosen command."""
import argparse
import logging
from common import setup_log


# 模块：命令编排；核心业务：否。
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "train", "serve"])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    setup_log(args.command)
    try:
        if args.command == "prepare":
            from prepare import prepare
            prepare()
        elif args.command == "train":
            from experiment import train_all
            train_all(args.epochs, args.device)
        else:
            from web import serve
            serve(args.port)
        logging.info("PROCESS_EXIT_CODE=0")
    except Exception:
        logging.exception("PROCESS_EXIT_CODE=1")
        raise


if __name__ == "__main__":
    main()
