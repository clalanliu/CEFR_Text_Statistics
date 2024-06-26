import os
# Ref: https://github.com/IlyaSemenov/wikipedia-word-frequency
file = os.path.join(os.path.dirname(__file__), 'enwiki-2023-04-13.txt')
with open(file, 'r', encoding='utf-8') as f:
    lines = f.readlines()
word_freq = {line.split(' ')[0]: float(line.split(' ')[1])/2474589909 for line in lines}
