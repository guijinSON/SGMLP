from transformers import AutoTokenizer

def load_tokenizer(HF_Name='roberta-base'):
    return  AutoTokenizer.from_pretrained(HF_Name)
