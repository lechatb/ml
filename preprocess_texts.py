# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 11:33:38 2023

@author: borovko

"""

import heapq
import math
import pprint as pp
import string
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
from zipfile import ZipFile

import numpy as np
import pymorphy2
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

stemmer = SnowballStemmer('russian')
stop_words = set(stopwords.words('russian'))
morph = pymorphy2.MorphAnalyzer()

lem_it = lambda x: morph.parse(x)[0].normal_form
stem_it = lambda x: stemmer.stem(x)


def get_idf_vocabulary(zipname='unigrams.cyr.lc.zip', filename='unigrams.cyr.lc', proc_it = lem_it) -> dict():
    d = defaultdict(int)
    try:
        with ZipFile(zipname, "r") as myzip:
            with myzip.open(filename) as file:
                lines = file.readlines()
                for i, l in enumerate(lines):
                    s = l.decode('utf-8')
                    k, a, b = s.split()
                    k = proc_it(k)
                    d[k] += int(a)
        return d
    except:
        return d


def preprocess_text(text, stemmer, stop_words, proc_it = lem_it):
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.lower().split()
    stemmed_words = [proc_it(word) for word in words]
    filtered_words = [word for word in stemmed_words if word not in stop_words]
    return ' '.join(filtered_words)

def preprocess_tfidf_texts(texts, proc_it = lem_it):
    """
    python которая принимает на вход numpy список текстов на русском языке, 
    исключает знаки препинания, приводит слова к нижнему регистру, 
    выполняет стеминг, делает из них словарь, исключает стоп-слова и 
    возвращает numpy список TFIDF encoders соответствующих этим текстам а также словарь
    """
    preprocessed_texts = [preprocess_text(text, stemmer, stop_words, proc_it=proc_it) for text in texts]
    vectorizer = TfidfVectorizer()
    tfidf_encoded = vectorizer.fit_transform(preprocessed_texts).toarray()
    vocabulary = vectorizer.vocabulary_
    return tfidf_encoded, vocabulary

def preprocess_count_texts(texts, proc_it = lem_it):
    """
    python которая принимает на вход numpy список текстов на русском языке, 
    исключает знаки препинания, приводит слова к нижнему регистру, 
    выполняет стеминг, делает из них словарь, исключает стоп-слова и 
    возвращает numpy список countvector encoders соответствующих этим текстам а также словарь
    """
    preprocessed_texts = [preprocess_text(text, stemmer, stop_words, proc_it=proc_it) for text in texts]
    vectorizer = CountVectorizer()
    count_encoded = vectorizer.fit_transform(preprocessed_texts).toarray()
    vocabulary = vectorizer.vocabulary_
    return count_encoded, vocabulary

def summarize_text(text: str, freq_dict: Dict[str, int], proc_it = lem_it) -> Tuple[str, List[str]]:
    
    # приведение текста к нижнему регистру и токенизация на слова
    words = word_tokenize(text.lower())
    
    # стемминг слов
    stemmed_words = [proc_it(word) for word in words]
    
    # подсчет количества вхождений каждого слова в текст
    word_freq = Counter(stemmed_words)
    
    # подсчет отношения частот слов в тексте к частотам слов в словаре
    word_tf_idf = {word: freq * math.log(len(freq_dict) / freq_dict[word]) \
                                         for word, freq in word_freq.items() \
                                         if word in freq_dict}    
    
    # Получаем 10 самых важных слов
    top_words = heapq.nlargest(10, word_tf_idf, key=word_tf_idf.get)
    
    # токенизация текста на предложения
    sentences = sent_tokenize(text)
    
    # подсчет отношения частот предложений в тексте к общему числу предложений
    sent_scores = {}
    for sent in sentences:
        score = 0
        sent_words = word_tokenize(sent.lower())
        stemmed_sent_words = [proc_it(word) for word in sent_words]
        for word in stemmed_sent_words:
            if word in word_tf_idf:
                score += word_tf_idf[word]
        sent_scores[sent] = score
    
    # выбор наиболее значимых предложений
    summary_sentences = heapq.nlargest(3, sent_scores, key=sent_scores.get)

    return " ".join(summary_sentences), top_words


if __name__ == "__main__":
    texts = np.array(["Уникальная информация", 
                  "Текст на русском языке, со знаками препинания!", 
                  "Другой текст на русском языке.",
                  "Если друг оказался вдруг и не друг, и не враг, а так..."])

    encoded_texts, vocabulary = preprocess_tfidf_texts(texts, proc_it=lem_it)
    
    print(encoded_texts)
    print(vocabulary)

    encoded_texts, vocabulary = preprocess_count_texts(texts)
    
    print(encoded_texts)
    print(vocabulary)
    voc = get_idf_vocabulary()
    txt = """
    Материалы ИноСМИ содержат оценки исключительно зарубежных СМИ и не отражают позицию редакции ИноСМИ
Читать inosmi.ru в
Запад желает России поражения, поэтому США и другие члены НАТО предприняли усилия, чтобы не допустить поддержку Москвы со стороны Китая, пишет MWM. Но у Пекина есть варианты укрепления российских вооруженных сил даже без прямых оружейных поставок.
19 февраля госсекретарь США Энтони Блинкен и посол при ООН Линда Томас-Гринфилд предостерегли Китай, что любые попытки поставить вооружения соседней России для Вашингтона — "красная линия". "Мы должны четко понимать, что если будут какие-либо мысли и усилия со стороны китайцев или других по оказанию смертоносной поддержки русским в их жестоком нападении на Украину, то это неприемлемо", — сказала Томас-Гринфилд в интервью телеканалу CNN, а госсекретарь Блинкен подчеркнул, что Вашингтон по-прежнему "крайне обеспокоен тем, что Китай рассматривает оказание смертоносной поддержки Москве". Блинкен добавил, что в беседе с верховным дипломатом Китая и членом Госсовета Ван И в кулуарах Мюнхенской конференции по безопасности он "явственно дал понять, что это чревато серьезными последствиями для двусторонних отношений", однако признал, что эту черту Китай "еще не пересек".
Читайте ИноСМИ в нашем канале в Telegram
За первый год российско-украинского конфликта Киев получил военную технику на десятки миллиардов долларов со всего западного мира, а также масштабную кадровую поддержку, разведданные и другие ресурсы. Помощь Запада варьируется: от сотен британских королевских морпехов, развернутых с апреля для операций с высокой степенью риска бок о бок с украинскими правительственными силами, до привлечения почти всей спутниковой сети НАТО для передачи ключевых разведданных, поддержки связи и целеуказания. The New York Times сообщила, что в эпицентре конфликта действует "невидимая сеть" ЦРУ. По данным газеты, США развернули на Украине "невидимую сеть коммандос и шпионов, чтобы как можно скорее предоставить ей оружие, разведданные и обучение". "Агенты ЦРУ тайно действуют в стране, в основном в столице, распределяя основную часть огромного массива разведданных, которыми США снабжают украинские силы", — отметила газета.

      """
    
    pp.pprint(summarize_text(txt, voc))