from pydantic import BaseModel
from torch import nn
import torch
from transformers import AutoModel

from src.Constant import PHOBERT_MODEL


class SentimentClassifier(nn.Module):
    def __init__(self, num_labels=3, layers_to_use=[6, 9, 10, 11]):
        super().__init__()
        self.num_labels = num_labels
        self.layers_to_use = layers_to_use
        self.phobert = AutoModel.from_pretrained(PHOBERT_MODEL, output_hidden_states=True)
        self.hidden_size = self.phobert.config.hidden_size
        self.dropout = nn.Dropout(0.1)

        # MLP với input là 4 * hidden_size (do ghép 4 lớp)
        self.classifier = nn.Sequential(
            nn.Linear(len(layers_to_use) * self.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states

        # Lấy các lớp đích và trích [CLS] token
        cls_embeddings = []
        for layer in self.layers_to_use:
            cls_embeddings.append(hidden_states[layer][:, 0, :])  # [CLS] ở vị trí 0

        # Ghép các embeddings
        concatenated = torch.cat(cls_embeddings, dim=1)
        concatenated = self.dropout(concatenated)

        # Đưa qua MLP
        logits = self.classifier(concatenated)
        return logits


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
