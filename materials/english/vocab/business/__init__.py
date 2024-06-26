import os
file = os.path.join(os.path.dirname(__file__), 'BSL_1.20_stats.csv')
with open(file, 'r', encoding='utf-8') as f:
    lines = f.readlines()[1:]

vocabs = {"business_1200": set([line.split(',')[0].strip() for line in lines if ',' in line])}
