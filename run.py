import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from parser import DocxParser
from chunker import TextChunker
from retriever import DenseRetrieverConfig, DenseRetriever
from embedding import HuggingFaceEmbedding
from llm import QwenChat
from tqdm import tqdm
import glob
import pandas as pd
import re
import os
import nltk
import time

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

llm_model = '/home/zbtrs/lz/models/DeepSeek-R1-0528-Qwen3-8B'
vector_model = '/home/zbtrs/lz/models/bge-large-zh-v1.5'

STORE_PATH = './index_store_bge'
DOCX_PATH = '/home/zbtrs/lz/初赛A榜/赛题制度文档'
COMP_DIR = '/home/zbtrs/lz/初赛A榜/数据集A'

bnb = True
think = False

OUTPUT_PATH = f"../results/{llm_model.split('/')[-1]}{'_bnb' if bnb else ''}{'_think' if think else ''}_doc8.json"

dp = DocxParser()
tc = TextChunker()
all_paragraphs = []
all_chunks = []

dense_config = DenseRetrieverConfig(
    model_name_or_path=vector_model,
    dim=1024,
    index_path=STORE_PATH
)
embedding_generator = HuggingFaceEmbedding(
    model_name=vector_model,
    device="cuda"
)

retriever = DenseRetriever(dense_config, embedding_generator)

if os.path.exists(STORE_PATH):
    retriever.load_index(STORE_PATH)
else:
    for filepath in tqdm(glob.glob(f'{DOCX_PATH}/*/*.docx')):
        paragraphs = dp.parse(filepath)
        all_paragraphs.append(paragraphs)
    for paragraphs in tqdm(all_paragraphs):
        chunks = tc.get_chunks(paragraphs, 256)
        all_chunks.extend(chunks)

    retriever.build_from_texts(all_chunks)
    retriever.save_index(STORE_PATH)

model = QwenChat(path=llm_model, device='cuda', bnb=bnb, think=think)

# 构建用户提问模板
def get_query(row):
    category = row['category']
    question = row['question']
    if category == '选择题':
        content = row['content']
        return f'{category}: {question}\n{content}'
    else:
        return f'{category}: {question}'

train = pd.read_json(f'{COMP_DIR}/train.json', lines=True)
test = pd.read_json(f'{COMP_DIR}/testA.json', lines=True)
train['query'] = train.apply(lambda row: get_query(row), axis=1)
test['query'] = test.apply(lambda row: get_query(row), axis=1)
train_ans = pd.read_json(f'{COMP_DIR}/train_answer.json', lines=True)
train['answer'] = train_ans['answer']

data = test
data = data.sort_values(by='category', key=lambda x: x != '选择题')

results = []
BATCH_SIZE = 4
for i in tqdm(range(0, len(data), BATCH_SIZE)):
    # 获取当前批次的数据
    batch = data.iloc[i:i+BATCH_SIZE]
    
    # 批量获取查询和内容
    queries = batch['query'].tolist()
    all_qtypes = batch['category'].tolist()
    all_contents = []
    for query in queries:
        contents = retriever.retrieve(query=query, top_k=8)
        content = '\n'.join(['- ' + content['text'] for content in contents])
        all_contents.append(content)
    
    # 批量推理
    batch_results, _ = model.chat(queries, all_qtypes, [], all_contents)
    
    # 如果返回单个结果（最后一批可能不足batch_size条）
    if isinstance(batch_results, str):
        results.append(batch_results)
    else:
        results.extend(batch_results)

data['answer_p'] = results

def postprocessing(text):
    result = text
    if '* 回答 *' in result:
        result = result.split('* 回答 *')[1]
    else:
        if '选择题' in result:
            if '</think>' in result:
                result = result.split('</think>')[1]
            else:
                result = 'C'
        else:
            if '</think>' in result:
                result = result.split('</think>')[1]
            else:
                result = result.replace('<think>', '')[-50:]
    return result.strip()
data['answer_p'] = data['answer_p'].apply(postprocessing)

res = []
for i, row in tqdm(data.iterrows()):
    ap = row['answer_p']
    if row['category'] == '选择题':
        res.append(list(map(lambda x: x.strip(), ap.split(','))))
    else:
        res.append(ap)

data['answer'] = res
data.sort_values(by='id')[['id', 'answer']].to_json(OUTPUT_PATH, lines=True, orient='records', force_ascii=False)

