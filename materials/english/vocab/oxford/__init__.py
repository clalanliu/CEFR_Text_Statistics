import os
import pandas as pd
file = os.path.join(os.path.dirname(__file__), 'oxford_5000.csv')
data = pd.read_csv(file)
vocabs = {
    'oxford_5000': set(data['word'].tolist()),
    'oxford_cefr_a1': set(data[data['cefr'] == 'a1']['word'].tolist()),
    'oxford_cefr_a2': set(data[data['cefr'] == 'a2']['word'].tolist()),
    'oxford_cefr_b1': set(data[data['cefr'] == 'b1']['word'].tolist()),
    'oxford_cefr_b2': set(data[data['cefr'] == 'b2']['word'].tolist()),
    'oxford_cefr_c1': set(data[data['cefr'] == 'c1']['word'].tolist()),
}
