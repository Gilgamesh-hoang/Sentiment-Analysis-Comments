import os

import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import pickle
from src.utils.Constant import PHOBERT_MODEL
from tqdm import tqdm

phobert = AutoModel.from_pretrained(PHOBERT_MODEL)
tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL, use_fast=False)


def get_phobert_vector(sentence: str) -> np.ndarray:
    """
    Convert a word-segmented Vietnamese sentence into a vector using PhoBERT.
    Args:
        sentence (str): The input sentence (must be word-segmented).
    Returns:
        np.ndarray: A fixed-size vector (shape: [768]) for training.
    """

    # Tokenize and convert to tensor
    input_ids = torch.tensor([tokenizer.encode(sentence, padding=True, max_length=256, truncation=True)])

    # Get embeddings from PhoBERT
    with torch.no_grad():
        features = phobert(input_ids)[0]  # Shape: [1, sequence_length, 768]

    # Apply mean pooling to get fixed-size vector
    sentence_embedding = features.mean(dim=1)  # Shape: [1, 768]

    return sentence_embedding.squeeze().numpy()  # Convert to NumPy array


def generate_vectors_dataset(input_dataset: str, output_folder: str, filename: str) -> None:
    """
    Convert a dataset of word-segmented Vietnamese sentences into vectors using PhoBERT
    and save them with their labels in a .pkl file.

    Args:
        input_dataset (str): The path to the input dataset.
        output_folder (str): The path to the output dataset.
        filename (str): The name of the output file.
    """

    if not os.path.exists(input_dataset):
        raise FileNotFoundError(f"File not found: {input_dataset}")

    vectors = []
    labels = []

    with open(input_dataset, "r", encoding="utf-8") as infile:
        for line in tqdm(infile, desc="Processing dataset"):
            parts = line.strip().split("\t")  # Dữ liệu phân tách bằng tab (\t)
            if len(parts) != 2:
                print(f"Dòng không hợp lệ: {line}")
                continue  # Bỏ qua dòng không hợp lệ

            text, label = parts  # Tách câu và nhãn
            vector = get_phobert_vector(text)  # Lấy vector từ câu

            vectors.append(vector)
            labels.append(label)

    # Lưu vectors và labels vào file .pkl
    # get file name without extension with '.' last index
    filename += '.pkl'
    output = os.path.join(output_folder, filename)
    with open(output, "wb") as f:
        pickle.dump((labels, vectors), f)

    print(f"Đã lưu {len(vectors)} vectors vào {output}")
