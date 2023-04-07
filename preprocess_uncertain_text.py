from fuzzywuzzy import process
from zipfile import ZipFile

# Задаем список правильных слов
correct_words = ['гроза', 'грозу', 'май', 'мая', 'люблю']


def load_dictionary(zipname='unigrams.cyr.lc.zip', filename='unigrams.cyr.lc'):
    correct_words = set()
    try:
        with ZipFile(zipname, "r") as myzip:
            with myzip.open(filename) as file:
                lines = file.readlines()
                for l in lines:
                    s = str(l.split(0))
                    correct_words.add(s.lower())
        return correct_words
    except:
        return correct_words


def correct_text(text, correct_words = correct_words):
    # Разбиваем текст на слова
    words = text.split()

    # Исправляем каждое слово в тексте с помощью fuzzywuzzy
    for i in range(len(words)):
        word = words[i]
        matches = process.extract(word, correct_words, limit=1)
        best_match = matches[0][0]
        if matches[0][1] < 70:
            continue    # Не исправляем слова, у которых нет достаточного соответствия
        words[i] = best_match

    # Возвращаем исправленный текст
    return ' '.join(words)