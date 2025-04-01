import argparse
import random
import numpy as np
import ruamel.yaml as yaml
import torch
import torch.backends.cudnn as cudnn

from dataset import create_dataset, create_loader
from models.person_search_model import ALBEF
from models.tokenization_bert import BertTokenizer
from utils.common import get_rank


def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config):
    model.train()

    for image1, image2, text1, text2, idx, replace in data_loader:
        # image1, image2 → (batch_size, C, H, W) where C is channels, H is height, W is width.
        # text1, text2   → (batch_size, seq_length)
        image1 = image1.to(device, non_blocking=True)
        image2 = image2.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)
        replace = replace.to(device, non_blocking=True)

        # text_input includes input_ids, token_type_ids, attention_mask
        # input_ids      → Encoded token IDs of the text.          | tensor([batch, seq_length])
        # token_type_ids → Differentiates sentences in a sentence pair (ignored).
        # attention_mask → Marks valid tokens (1) and padding (0). | tensor([batch, seq_length])
        text_input1 = tokenizer(text1, padding="longest", max_length=config["max_words"], return_tensors="pt").to(
            device
        )
        text_input2 = tokenizer(text2, padding="longest", max_length=config["max_words"], return_tensors="pt").to(
            device
        )
        alpha = 0.4
        model(image1, image2, text_input1, text_input2, alpha=alpha, idx=idx, replace=replace)


def seed_everything(seed):
    # Fix the seed for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True


def main(args, config):
    device = torch.device(args.device)

    seed = args.seed + get_rank()
    seed_everything(seed)

    train_dataset = create_dataset("ps", config)
    train_loader = create_loader(train_dataset, config["batch_size_train"])

    max_epoch = config["schedular"]["epochs"]
    warmup_steps = config["schedular"]["warmup_epochs"]
    start_epoch = 0
    best = 0
    best_epoch = 0
    best_log = ""

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
    model = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer)
    model = model.to(device)

    # Optimizer and learning rate scheduler
    # arg_opt = utils.AttrDict(config["optimizer"])
    # optimizer = create_optimizer(arg_opt, model)
    # arg_sched = utils.AttrDict(config["schedular"])
    # lr_scheduler, _ = create_scheduler(arg_sched, optimizer)

    # train
    _ = ""
    train_stats = train(model, train_loader, _, tokenizer, _, _, device, _, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/base.yaml")
    parser.add_argument("--text_encoder", default="bert-base-uncased")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    yaml_loader = yaml.YAML()
    with open(args.config, "r") as file:
        config = yaml_loader.load(file)

    main(args, config)
