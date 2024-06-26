import os
file = os.path.join(os.path.dirname(__file__), 'tw_senior_high.txt')
with open(file, 'r', encoding='utf-8') as f:
    lines = f.readlines()[1:]

vocabs = {"tw_senior_high": set([line.strip().split(' ')[0].strip().strip('*')
                                for line in lines if len(line.strip()) > 1])}
