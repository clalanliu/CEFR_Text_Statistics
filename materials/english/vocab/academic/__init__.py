import os
file = os.path.join(os.path.dirname(__file__), 'NAWL_1.2_stats.csv')
with open(file, 'r', encoding='utf-8') as f:
    lines = f.readlines()[1:]

vocabs = {"academic_1200": set([line.split(',')[0].strip() for line in lines if ',' in line])}
