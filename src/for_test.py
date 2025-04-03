import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from src.Constant import PHOBERT_MODEL

LABEL_MAP = {
    'Negative': 0, 'Neutral': 1, 'Positive': 2
}

class CustomPhoBERT(nn.Module):
    def __init__(self, num_labels, layers_to_use=[6, 9, 10, 11]):
        super().__init__()
        self.num_labels = num_labels
        self.layers_to_use = layers_to_use
        self.phobert = AutoModel.from_pretrained(PHOBERT_MODEL, output_hidden_states=True)
        self.hidden_size = self.phobert.config.hidden_size
        self.dropout = nn.Dropout(0.1)

        # MLP với input là 4 * hidden_size (do ghép 4 lớp)
        self.classifier = nn.Sequential(
            nn.Linear(4 * self.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
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


def predict(text: str, tokenizer, device, model, max_len=250) -> str:
    encoded_review = tokenizer.encode_plus(
        text,
        max_length=max_len,
        truncation=True,
        add_special_tokens=True,
        padding='max_length',
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors='pt',
    )

    input_ids = encoded_review['input_ids'].to(device)
    attention_mask = encoded_review['attention_mask'].to(device)

    with torch.no_grad():
        output = model(input_ids, attention_mask)
        _, y_pred = torch.max(output, dim=1)

    # Extract the scalar value from the tensor
    y_pred_value = y_pred.item()  # Convert tensor to Python integer

    if y_pred_value == 0:
        return 'Negative'
    elif y_pred_value == 1:
        return 'Neutral'
    else:
        return 'Positive'


if __name__ == '__main__':
    print('Loading model...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomPhoBERT(num_labels=3)  # Số lớp phải giống lúc huấn luyện
    model.load_state_dict(
        torch.load('/Model/phobert_text_classification.pth', map_location=device))
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL, use_fast=False)

    with open('E:\\Sentiment-Analysis-Comments\\Dataset\\test_data1.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line, label = line.split('\t')
            sentiment = predict(line, tokenizer, device, model)
            print(f'{line.strip()} - {sentiment}')

