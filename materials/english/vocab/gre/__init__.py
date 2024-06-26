import os
file = os.path.join(os.path.dirname(__file__), 'gre.csv')
with open(file, 'r', encoding='utf-8') as f:
    lines = f.readlines()[1:]

vocabs = {"gre_6000": set([line.split(',')[0].strip() for line in lines])}
