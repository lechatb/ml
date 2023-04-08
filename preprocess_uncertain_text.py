from fuzzywuzzy import process, fuzz
from zipfile import ZipFile

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
correct_words = load_dictionary()


def correct_text(text, dictionary = correct_words):
    """
    Corrects a given text string using a dictionary of valid words.
    """
    # Split text into individual words
    words = text.split()
    
    # Loop over each word and try to match to a word in the dictionary
    for i, word in enumerate(words):
        match_scores = [(fuzz.ratio(word, dict_word), dict_word) for dict_word in dictionary]
        best_match = max(match_scores, key=lambda x: x[0])
        
        # If the best match score is above threshold, replace the word with the dictionary word
        if best_match[0] >= 70:
            words[i] = best_match[1] #??? почему не 0?

    return " ".join(words)


from numba import njit, int32, types
import numpy as np

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

            d[i][j] = min(d[i - 1][j] + 1, # Deletion
                          d[i][j - 1] + 1, # Insertion
                          d[i - 1][j - 1] + cost) # Substitution

    return d[n][m]

@njit(int32(types.unicode_type, types.unicode_type))
def edit_distance_numba(s, t):
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

            d[i][j] = min(d[i - 1][j] + 1, # Deletion
                          d[i][j - 1] + 1, # Insertion
                          d[i - 1][j - 1] + cost) # Substitution

    return d[n][m]


#print(correct_text('формации © временном испопнении обязанностей я китронёра Ирафучастчикы эынка'))
print(edit_distance_numba('городок', 'гоoрдок'))

import timeit

print(timeit.timeit(lambda: edit_distance_numba('городок', 'гоoрдок'), number=10000))
print(timeit.timeit(lambda: edit_distance('городок', 'гоoрдок'), number=10000))