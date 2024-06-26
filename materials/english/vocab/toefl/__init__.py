import os
file = os.path.join(os.path.dirname(__file__), 'raw.txt')
with open(file, 'r', encoding='utf-8') as f:
    lines = f.readlines()
vocabs = {'toefl_5500': set([line.split('#')[0].strip() for line in lines if '#' in line])}
