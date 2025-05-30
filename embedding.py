import warnings
warnings.simplefilter('ignore')
import numpy as np
import pandas as pd
import os
import re
import sys
import json
import glob
import copy
import math
import string
from io import BytesIO
from collections import Counter
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Any
import faiss
import datrie
from tqdm import tqdm
from docx import Document
from hanziconv import HanziConv
from dataclasses import dataclass
from nltk import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class EmbeddingGenerator(ABC):
    """
    Abstract base class for embedding generators.
    All embedding models should inherit from this class and implement the required methods.
    """

    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts (List[str]): List of text strings to generate embeddings for

        Returns:
            np.ndarray: Array of embeddings with shape (len(texts), embedding_dimension)
        """
        pass
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text (str): Text string to generate embedding for

        Returns:
            np.ndarray: Embedding vector with shape (embedding_dimension,)
        """
        return self.generate_embeddings([text])[0]
    @staticmethod
    def cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vector1 (np.ndarray): First vector
            vector2 (np.ndarray): Second vector

        Returns:
            float: Cosine similarity score between 0 and 1
        """
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        return 0 if not magnitude else dot_product / magnitude
class HuggingFaceEmbedding(EmbeddingGenerator):
    def __init__(
            self,
            model_name: str,
            device: str = None,
            trust_remote_code: bool = True
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**encoded_input)
            embeddings = outputs[0][:, 0]  # Use CLS token embeddings
        normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return normalized_embeddings.cpu().numpy()