from pydantic import BaseModel
from torch import nn
import torch
from transformers import AutoModel

from src.util.Constant import PHOBERT_MODEL


class SentimentClassifier(nn.Module):
    def __init__(self, num_labels=3, layers_to_use=[6, 9, 10, 11], dropout_prob=0.4):
        super().__init__()
        self.num_labels = num_labels
        self.layers_to_use = layers_to_use
        self.phobert = AutoModel.from_pretrained(PHOBERT_MODEL, output_hidden_states=True)
        self.hidden_size = self.phobert.config.hidden_size
        self.dropout = nn.Dropout(dropout_prob)

        for name, param in self.phobert.named_parameters():
            # Kiểm tra nếu tham số thuộc các layer transformer
            if "layer" in name:
                # Lấy số thứ tự layer
                layer_num = int(name.split("layer.")[1].split(".")[0])
                # Đóng băng nếu layer không nằm trong layers_to_use
                if layer_num not in self.layers_to_use:
                    param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(len(layers_to_use) * self.hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.4),

            nn.Linear(256, num_labels)
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
