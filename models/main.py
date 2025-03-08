from xbert import BertConfig, BertForMaskedLM

if __name__ == '__main__':
    text_encoder = 'bert-base-uncased'
    bert_config = BertConfig.from_json_file('./models/config_bert.json')

    bert_model = BertForMaskedLM.from_pretrained(
        text_encoder, config=bert_config)
