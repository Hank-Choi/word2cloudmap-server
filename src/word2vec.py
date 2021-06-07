from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
import pandas as pd
import numpy as np


def search(name, key):
    loaded_model = KeyedVectors.load_word2vec_format(name)
    return list(map(lambda x: {'text': x[0], 'value': x[1]},loaded_model.most_similar(key, topn=100)))


def train(sentences, model_name):
    model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=5, workers=4, sg=1)
    model.wv.sort_by_descending_frequency()
    model.wv.save_word2vec_format(model_name)  # 모델 저장
    # print(model.wv.key_to_index)
    top_words = model.wv.index_to_key
    count = model.wv.expandos['count']
    return list(map(lambda x: {'text': x[0], 'value': int(x[1])}, zip(top_words,count)))


def preprocess(article):
    # 입력 코퍼스에 대해서 NLTK를 이용하여 문장 토큰화를 수행.
    sent_text = sent_tokenize(article)

    df = pd.DataFrame({'document': sent_text})
    # 특수 문자 제거
    df['clean_doc'] = df['document'].str.replace("[^a-zA-Z]", " ")
    # 길이가 3이하인 단어는 제거 (길이가 짧은 단어 제거)
    df['clean_doc'] = df['clean_doc'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 2]))
    # 전체 단어에 대한 소문자 변환
    df['clean_doc'] = df['clean_doc'].apply(lambda x: x.lower())

    df.replace("", float("NaN"), inplace=True)
    df.dropna(inplace=True)
    stop_words = stopwords.words('english')
    tokenized_doc = df['clean_doc'].apply(lambda x: x.split())
    tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])
    # 단어 s 제거
    tokenized_doc = tokenized_doc.apply(lambda x: [item[:-1] if item[-1] == 's' else item for item in x])
    tokenized_doc = tokenized_doc.to_list()
    drop_train = [index for index, sentence in enumerate(tokenized_doc) if len(sentence) <= 1]
    tokenized_doc = np.delete(tokenized_doc, drop_train, axis=0)

    return tokenized_doc


if __name__ == '__main__':
    with open("test.txt", "r") as myfile:
        data = myfile.read()
        print(train(preprocess(data), "test"))
        print(search("test", "emma"))
