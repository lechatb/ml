import re
import string
import timeit
import numpy as np
from numba import njit, int32, types, jit, prange
from fuzzywuzzy import process, fuzz
from zipfile import ZipFile
import numba as nb


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


def correct_text(text, dictionary=correct_words):
    """
    Corrects a given text string using a dictionary of valid words.
    """
    # Split text into individual words
    words = text.split()

    # Loop over each word and try to match to a word in the dictionary
    for i, word in enumerate(words):
        match_scores = [(fuzz.ratio(word, dict_word), dict_word)
                        for dict_word in dictionary]
        best_match = max(match_scores, key=lambda x: x[0])

        # If the best match score is above threshold, replace the word with the dictionary word
        if best_match[0] >= 70:
            words[i] = best_match[1]

    return " ".join(words)


def edit_distance(s, t):
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


#@njit(int32(types.unicode_type, types.unicode_type))
@jit
def edit_distance_numba(s: types.string, t:types.string):
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

#@nb.njit(nb.types.Tuple((nb.types.int32, nb.types.List(nb.types.unicode_type)))(nb.types.unicode_type))
#@nb.njit(nogil=True, cache=False)
def check_word(word: str) -> tuple:
    global correct_words
    dictionary=correct_words

    dist = list(map(lambda x: edit_distance_numba(word, x), dictionary))
    dist = np.asarray(dist)
    min_dist=np.min(dist)
    res = dictionary[np.argmin(dist)]
    return min_dist, res


#print(correct_text('формации © временном испопнении обязанностей я китронёра Ирафучастчикы эынка'))
print(edit_distance_numba('городок', 'гоoрдок'))


print(timeit.timeit(lambda: edit_distance_numba('городокккк', 'гоoрдокккк'), number=10000))
print(timeit.timeit(lambda: edit_distance('городокккк', 'гоoрдок'), number=10000))

@njit
def correct_text(input_string, fix_hyphenation=True):
    def split_string(string):
        return re.findall(r"[\w']+|[ -.,!?;]", string)

    def find_word_left(arr):
        for i in range(len(arr)):
            if isinstance(arr[i], str) and arr[i].isalpha():
                return i
        return -1  # если такой ячейки не найдено

    def find_word_right(arr):
        for i in range(-1, -len(arr), -1):
            if isinstance(arr[i], str) and arr[i].isalpha():
                return i
        return -1  # если такой ячейки не найдено

    result_string = (input_string.split('''\n'''))
    result_string = list(map(split_string, result_string))
    if fix_hyphenation:
        for i in range(len(result_string)-1):
            i1 = find_word_right(result_string[i])
            i2 = find_word_left(result_string[i+1])
            if i1 == -1 or i2 == -1:
                continue
            w1 = result_string[i][i1]
            w2 = result_string[i+1][i2]
            w3 = w1+w2
            a1=check_word(w1)
            a2=check_word(w2)
            a3=check_word(w3)
            
            if a1[0] <= a2[0] + a3[0]:
                result_string[i][i1]=w3
                result_string[i]=result_string[i][:i1+1]
                result_string[i+1][i2]=result_string[i+1][i2+1:]

    return result_string


print( check_word('Дшм'))

s = '''о Бедном? Гусаре! Замолвите Сло-
во, Ваш Муж Не Пускает Меня/ На Постой
'''
#print(correct_text(s))


