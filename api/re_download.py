from transformers import AutoTokenizer
import os

tokenizer_name = os.getenv("TOKENIZER_NAME", "facebookAI/xlm-roberta-base")

chunker_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)