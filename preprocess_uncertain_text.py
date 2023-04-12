from functools import partial
import re
import timeit
import numpy as np
from numba import njit, types
from zipfile import ZipFile
import multiprocessing as mp


def load_dictionary(zipname='unigrams.cyr.lc.zip', filename='unigrams.cyr.lc'):
    correct_words = set()
    try:
        with ZipFile(zipname, "r") as myzip:
            with myzip.open(filename) as file:
                lines = file.readlines()
                for l in lines:
                    l = l.decode('utf-8')
                    s = str(l.split()[0])
                    correct_words.add(s.lower())
        return list(correct_words)
    except:
        return list()


# Задаем список правильных слов
correct_words = np.array(load_dictionary())


def edit_distance(s, t):  # Оставил для сравнения быстродействия с версией numba
    n = len(s)
    m = len(t)

    # Create matrix of zeros
    d = [[0] * (m + 1) for _ in range(n + 1)]

    # Initialize matrix
    for i in range(n + 1):
        d[i][0] = i

    for j in range(m + 1):
        d[0][j] = j

    # Calculate edit distance
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if s[i - 1] == t[j - 1]:
                cost = 0
            else:
                cost = 1

            d[i][j] = min(d[i - 1][j] + 1,  # Deletion
                          d[i][j - 1] + 1,  # Insertion
                          d[i - 1][j - 1] + cost)  # Substitution

    return d[n][m]


@njit(fastmath=True)
def edit_distance_numba(s: types.unicode_type, t: types.unicode_type):
    n = len(s)
    m = len(t)

    # Create matrix of zeros
    d = np.zeros((n+1, m+1), dtype=np.int32)

    # Initialize matrix
    for i in range(n + 1):
        d[i][0] = i

    for j in range(m + 1):
        d[0][j] = j

    # Calculate edit distance
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if s[i - 1] == t[j - 1]:
                cost = 0
            else:
                cost = 1

            d[i][j] = min(d[i - 1][j] + 1,  # Deletion
                          d[i][j - 1] + 1,  # Insertion
                          d[i - 1][j - 1] + cost)  # Substitution

    return d[n][m]


def check_word(word) -> tuple:
    dist = list(map(partial(edit_distance_numba, t=word), correct_words))
    min_dist = np.min(dist)
    res = correct_words[np.where(dist == min_dist)[0]]
    return (min_dist, word, res)


def check_words(words: list, cpu_cnt=mp.cpu_count()):
    with mp.get_context("spawn").Pool(cpu_cnt) as p:
        res = list(p.map(check_word, words))
    return res


def correct_text(input_string):

    def split_string(text):
        return re.findall(r'\w+', text)

    result_string = split_string(input_string.lower())

    double_string = [result_string[i] + result_string[i+1]
                     for i in range(len(result_string)-1)]

    result_string = check_words(result_string)
    double_string = check_words(double_string)

    processed_string = list()
    i = 0
    while i < len(double_string):
        if double_string[i][0] <= result_string[i][0]+result_string[i+1][0]:
            processed_string.append(double_string[i])
            i += 1
        else:
            processed_string.append(result_string[i])
            if i == len(double_string)-1:
                processed_string.append(result_string[i+1])
        i += 1

    return processed_string


if __name__ == '__main__':
    print(edit_distance_numba('городок', 'гоoрдок'))

    print("Время на 10000 сравнений с нумба", timeit.timeit(lambda: edit_distance_numba(
        'городокккк', 'гоoрдок'), number=10000))
    print("Время на 10000 сравнений без нумба", timeit.timeit(lambda: edit_distance(
        'городокккк', 'гоoрдок'), number=10000))

    print("Время на 1 корректуру слова", timeit.timeit(lambda: check_word('гоoрдок'), number=10)/10)
    
    s = '''о Бетном? Гусаре! Замулвите Сло-
    во, Ваш Муж Не Пус кает Меня/ На Постой
    '''
    def func():
        print(correct_text(s))
    print("Время на отработку текста:", timeit.timeit(func, number=1))
