import argparse
import ruamel.yaml as yaml
from torch import nn

from dataset import create_dataset
from models.tokenization_bert import BertTokenizer
from models.xbert import BertConfig, BertForMaskedLM


def main(args, config):
    train_dataset = create_dataset("ps", config)

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
    # # image1, image2 → (batch_size, C, H, W) where C is channels, H is height, W is width.
    # # text1, text2 → (batch_size, seq_length)
    text1 = [
        "a pedestrian male wearing a short white down jacket mobile phone in hand bag slung over his shoulder no glasses and brown trousers",
        "the woman in hat wears a red down jacket tight black pants and a pair of black leather boots and carries a dark blue backpack",
        "he is a man he has a black jacket and hat his jeans are blue and his shoes are brown he is walking",
        "a short hair man is walking he wears white slacks and a dark blue down jacket he carries a black bag",
        "the woman is wearing a black down jacketa pair of black leggings and a pair of black low heeled shoesshe is weairng a army green hat and a black mask and there is a plastic bag in her right hand",
        "a young women is wearing a black overcoat a white hat and a grey glove she is carrying a deep blue bag and wearing a pair of black shoes",
        "the walking female with glasses wears a red coat and a black scarfthe pedestrian under a grey hat has her black scarf and red coat on",
        "a man with short black hair was walking in a grey and blue down jacket and light grey trousers",
        "a woman was walking in a red coat and black leggings she had long black hair and a flat expression she was carrying a black bag with her hands in her pocket",
        "a foreign woman with long blond hair is walking down the road in a long black down jacket black tights and black sneakers",
        "a middle aged woman with short hair wearing a red down jacket and black trousers she also wears a pair of black shoes",
        "the woman is seen from the side she is wearing a black pie blue skinny jeans and a pair of brown snow boots",
        "a middle aged man was wearing a brown coat black trousers and red black shoes he is turning to the left",
    ]
    text2 = [
        "a man looking down at the ground while walking wears a brown jacket with hood and a pair of dark trousers",
        "the woman in hat wears a red down jacket tight black pants and a pair of black leather boots and carries a dark blue backpack",
        "he is a man he has a black jacket and hat his jeans are blue and his shoes are brown he is walking",
        "a short hair man is walking he wears white slacks and a dark blue down jacket he carries a black bag",
        "there is a man wearing a black down jacket black pants and black shoes he is also wearing a brown hat and he is carrying a white plastic bag in his right hand",
        "a middle aged woman in a blue knitted hat with a pair of glasses wears grey gloves and a kind snow boots while carrying a tote bag",
        "the walking female with glasses wears a red coat and a black scarfthe pedestrian under a grey hat has her black scarf and red coat on",
        "a man with short black hair was walking in a grey and blue down jacket and light grey trousers",
        "a woman was walking in a red coat and black leggings she had long black hair and a flat expression she was carrying a black bag with her hands in her pocket",
        "the woman is walking she wears a long ponytail a pair of black earphones a black down coat black leggings and black shoes she holds a green bag in her right hand",
        "a middle aged woman with short hair wearing a red down jacket and black trousers she also wears a pair of black shoes",
        "the woman is seen from the side she is wearing a black pie blue skinny jeans and a pair of brown snow boots",
        "a middle aged man was wearing a brown coat black trousers and red black shoes he is turning to the left",
    ]
    # image1 = image1.to(device, non_blocking=True)
    # image2 = image2.to(device, non_blocking=True)
    # idx = idx.to(device, non_blocking=True)
    # replace = replace.to(device, non_blocking=True)
    # text_input1 = tokenizer(text1, padding='longest', max_length=config['max_words'], return_tensors="pt").to(device)
    # text_input2 = tokenizer(text2, padding='longest', max_length=config['max_words'], return_tensors="pt").to(device)
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
