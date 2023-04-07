import csv
import pprint as pp
import re
from collections import defaultdict

import feedparser
import pandas as pd
from bs4 import BeautifulSoup

newsurls = {'Kommersant': 'https://www.kommersant.ru/RSS/news.xml',
            'Lenta.ru': 'https://lenta.ru/rss/',
            'Vesti': 'https://www.vesti.ru/vesti.rss',
            'news.ru':'https://news.ru/rss/',
            'finam1':'https://www.finam.ru/analysis/conews/rsspoint/',
            'finam_world':'https://www.finam.ru/international/advanced/rsspoint/'} 

f_all_news = 'allnews23march.csv'
f_certain_news = 'certainnews23march.csv'

vector1 = 'ДолЛАР|РубЛ|ЕвРО' #пример таргетов
vector2 = 'ЦБ|СбЕРбАНК|курс'

def download_rss( rss_set = newsurls) -> pd.DataFrame:
    rss_dict = defaultdict(list)
    for _, rss_url in rss_set.items():
        feed = feedparser.parse( rss_url)
        for field in ['title',
                      'description',
                      'link',
                      'published',
                      'summary']:
            rss_dict[field] += [BeautifulSoup(i[field]).get_text() for i in feed['items']]
    df = pd.DataFrame.from_dict(rss_dict)
    return df


def looking_for_certain_news(all_news_filepath, certain_news_filepath, target1, target2): 
    #функция для поиска, а затем записи
    #определенных новостей по таргета,
    #затем возвращает этот датасет
    df = pd.read_csv(all_news_filepath)
    
    result = df.apply(lambda x: x.str.contains(target1, na=False,
                                    flags = re.IGNORECASE, regex=True)).any(axis=1)
    result2 = df.apply(lambda x: x.str.contains(target2, na=False,
                                    flags = re.IGNORECASE, regex=True)).any(axis=1)
    new_df = df[result&result2]
        
    new_df.to_csv(certain_news_filepath
                     ,sep = '\t', encoding='utf-8-sig')
        
    return new_df

if __name__ == '__main__':
    print(download_rss())