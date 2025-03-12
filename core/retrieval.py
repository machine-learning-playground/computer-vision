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

    # train
    model.train()

    for image1, image2, text1, text2, idx, replace in train_loader:
        image1 = image1.to(device, non_blocking=True)
        image2 = image2.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)
        replace = replace.to(device, non_blocking=True)
        text_input1 = tokenizer(text1, padding="longest", max_length=config["max_words"], return_tensors="pt").to(
            device
        )
        text_input2 = tokenizer(text2, padding="longest", max_length=config["max_words"], return_tensors="pt").to(
            device
        )
        alpha = 0.4
        model(image1, image2, text_input1, text_input2, alpha=alpha, idx=idx, replace=replace)


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
