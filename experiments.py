import re

text = '''Привет, мир! 
Как дела?'''

# Разделение текста на слова и знаки препинания
tokens = re.split(r'(\W+)', text)

# Вывод токенов
print(tokens)