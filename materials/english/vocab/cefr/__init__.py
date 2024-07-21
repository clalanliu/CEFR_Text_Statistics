import os
import pandas as pd
file = os.path.join(os.path.dirname(__file__), 'words_data.csv')
data = pd.read_csv(file)
vocabs = {
    'cefr_a1': set(data[data['Level'] == 'A1']['Word'].tolist()),
    'cefr_a2': set(data[data['Level'] == 'A2']['Word'].tolist()),
    'cefr_b1': set(data[data['Level'] == 'B1']['Word'].tolist()),
    'cefr_b2': set(data[data['Level'] == 'B2']['Word'].tolist()),
    'cefr_c1': set(data[data['Level'] == 'C1']['Word'].tolist()),
    'cefr_c2': set(data[data['Level'] == 'C2']['Word'].tolist()),
}
