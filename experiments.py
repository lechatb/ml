import pymorphy2
from nltk.stem.snowball import SnowballStemmer
import timeit

morph = pymorphy2.MorphAnalyzer()
stemmer = SnowballStemmer('russian')

lem_it = lambda x: morph.parse(x)[0].normal_form
stem_it = lambda x: stemmer.stem(x)
none_it = lambda x: x

arr=['Прогуливаешься' for i in range(1000000)]
print(len(arr))
print( lem_it('Прогуливаешься'))
print( stem_it('Прогуливаешься'))

print(timeit.timeit(lambda: map(lem_it, arr), number=1))
print(timeit.timeit(lambda:stem_it('Прогуливаешься'), number=500_000))
print(timeit.timeit(lambda: [lem_it('Прогуливаешься') for i in range(1000000)], number=1))
