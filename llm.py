import os
os.environ['TRANSFORMERS_VERBOSITY']='error'
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
from template import *

class BaseModel:
    def __init__(self, path: str = '') -> None:
        self.path = path
    def chat(self, prompts: Union[str, List[str]], history: List[dict], content: Union[str, List[str]]) -> Union[str, List[str]]:
        pass
    def load_model(self):
        pass

class QwenChat(BaseModel):
    def __init__(self, path: str = '', device: str = 'cuda', bnb: bool = False, think: bool = False) -> None:
        super().__init__(path)
        self.device = device
        self.load_model(bnb)
        self.think = think

    def chat(self, queries: Union[str, List[str]], question_type: Union[str, List[str]], history: List = [], content: Union[str, List[str]] = '') -> tuple[List[str], Any]:
        # 确保输入是列表形式
        if isinstance(queries, str):
            queries = [queries]
        if isinstance(content, str):
            content = [content] * len(queries)
        
        # 确保查询和内容列表长度匹配
        assert len(queries) == len(content), "查询数量必须与内容数量相匹配"
        
        # 构建所有消息
        all_texts = []
        for query, doc, qtype in zip(queries, content, question_type):
            prompt = USER_PROMPT_TEMPLATE.replace('###DOCUMENT###', doc).\
                                        replace('###QUESTION###', query)
            messages = [
                {"role": "system", "content": XZT_SPROMPT if qtype == '选择题' else WDT_SPROMPT},
                {"role": "user", "content": prompt}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.think
            )
            all_texts.append(text)

        # 批量处理所有输入
        
        model_inputs = self.tokenizer(all_texts, return_tensors="pt", padding=True).to(self.device)
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=1024,
            do_sample=False,
            top_k=10
        )
        
        # 提取生成的新token
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        # 解码所有响应
        responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        # 如果输入是单个字符串，返回单个响应
        if len(queries) == 1:
            return responses[0], history
            
        return responses, history
    def load_model(self, bnb: bool):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True, padding_side="left")
        if bnb:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.path,
                quantization_config=bnb_config,
                attn_implementation="flash_attention_2",
                device_map=self.device,
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map=self.device,
                trust_remote_code=True
            )
        print("load model success")