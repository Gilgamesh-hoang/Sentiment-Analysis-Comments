import torch
from transformers import AutoTokenizer
import os
from src.label_encoder import decoder
from src.type import SentimentClassifier
import logging

from src.Constant import PHOBERT_MODEL

class ClassificationService:
    def __init__(self):
        self.__model = None
        self.__device = None
        self.__tokenizer = None
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(levelname)s - [in %(name)s:%(funcName)s():%(lineno)d] - %(message)s')
        logging.info('Service initialized')

    def load_components(self, model_path):
        if os.path.exists(model_path) is False:
            raise FileNotFoundError(f'Model file not found at {model_path}')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SentimentClassifier(n_classes=8)  # Số lớp phải giống lúc huấn luyện
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        self.__device = device
        self.__model = model
        self.__tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL, use_fast=False)
        logging.info('Model loaded successfully')

    def predict(self, text: str, max_len=250)->str:
        encoded_review = self.__tokenizer.encode_plus(
            text,
            max_length=max_len,
            truncation=True,
            add_special_tokens=True,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt',
        )

        input_ids = encoded_review['input_ids'].to(self.__device)
        attention_mask = encoded_review['attention_mask'].to(self.__device)

        with torch.no_grad():
            output = self.__model(input_ids, attention_mask)
            _, y_pred = torch.max(output, dim=1)

        # Extract the scalar value from the tensor
        y_pred_value = y_pred.item()  # Convert tensor to Python integer

        return decoder(y_pred_value)  # Pass the integer to decoder