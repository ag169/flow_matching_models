import argparse
from lib.train_lib import Trainer


def main():
    parser = argparse.ArgumentParser(
        description="Train conditional flow matching model"
    )
    parser.add_argument("--config", type=str, help="Path to train config file")
    parser.add_argument("--ckpt_dir", type=str, help="Path to checkpoint directory")

    args = parser.parse_args()

    trainer = Trainer(config_path=args.config, ckpt_dir_path=args.ckpt_dir)
    trainer.train()

    print("-" * 80)
    print("Finished training!")


if __name__ == "__main__":
    main()
