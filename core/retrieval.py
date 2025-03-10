import argparse
import ruamel.yaml as yaml
from torch import nn

from models.tokenization_bert import BertTokenizer
from models.xbert import BertConfig, BertForMaskedLM


def main(args, config):
    text_encoder_mode = args.text_encoder
    embed_dim = config["embed_dim"]  # out_features: 256

    tokenizer = BertTokenizer.from_pretrained(text_encoder_mode)
    bert_config = BertConfig.from_json_file(config["bert_config"])
    text_encoder = BertForMaskedLM.from_pretrained(text_encoder_mode, config=bert_config)

    # in_features: 768 (must be a multiple of the number of attention heads (12))
    text_width = text_encoder.config.hidden_size
    text_proj = nn.Linear(text_width, embed_dim)  # 768 → 256

    # # forward()
    # # extract text features
    # text_output = text_encoder.bert(
    #     text2.input_ids, attention_mask=text2.attention_mask, return_dict=True, mode="text"
    # )
    # text_embeds = text_output.last_hidden_state
    # text_feat = F.normalize(text_proj(text_embeds[:, 0, :]), dim=-1)

    print("text_width: ", text_width, text_proj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/base.yaml")
    parser.add_argument("--text_encoder", default="bert-base-uncased")
    args = parser.parse_args()

    yaml_loader = yaml.YAML()
    with open(args.config, "r") as file:
        config = yaml_loader.load(file)

    main(args, config)
