from pydantic import BaseModel
from torch import nn
from transformers import AutoModel

from src.Constant import PHOBERT_MODEL


class SentimentClassifier(nn.Module):
    def __init__(self, n_classes=8):
        super(SentimentClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(PHOBERT_MODEL)
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        last_hidden_state, output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False  # Dropout will errors if without this
        )

        x = self.drop(output)
        x = self.fc(x)
        return x


class Response:

    def __init__(self, message: str = None, data: str = None):
        self.message = message
        self.data = data

    def to_dict(self):
        return {
            "message": self.message,
            "data": self.data.upper()
        }


class SentimentRequest(BaseModel):
    content: str
