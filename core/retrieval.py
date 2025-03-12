import argparse
import ruamel.yaml as yaml
import torch

from dataset import create_dataset, create_loader
from models.person_search_model import ALBEF
from models.tokenization_bert import BertTokenizer


def main(args, config):
    device = torch.device(args.device)
    train_dataset = create_dataset("ps", config)
    train_loader = create_loader(train_dataset, config["batch_size_train"])

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
    model = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer)
    model = model.to(device)
    model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/base.yaml")
    parser.add_argument("--text_encoder", default="bert-base-uncased")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    yaml_loader = yaml.YAML()
    with open(args.config, "r") as file:
        config = yaml_loader.load(file)

    main(args, config)
