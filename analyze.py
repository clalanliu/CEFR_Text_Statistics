from materials.utils import get_word_freq, match_word_with_banks
import numpy as np
import datasets
import pandas as pd
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))


def lemmatize(sentence):
    # Tokenize the sentence into words
    words = nltk.word_tokenize(sentence.lower())
    pos_tags = nltk.pos_tag(words)
    lemmatized = [
        (
            WordNetLemmatizer().lemmatize(word, pos[0].lower())
            if pos[0].lower() in ["a", "n", "v"]
            else WordNetLemmatizer().lemmatize(word)
        )
        for word, pos in pos_tags
    ]
    return lemmatized


def add_word_freq(e):
    freqs = [get_word_freq(w) for w in e['lemmatized_clean']]
    e['word_freq_mean'] = np.mean(freqs)
    e['word_freq_75'] = np.percentile(freqs, 75)
    e['word_freq_25'] = np.percentile(freqs, 25)
    return e


def add_oxford(e):
    matched = [match_word_with_banks(w) for w in e['lemmatized_clean']]
    e['cefr_A1'] = sum(
        [1 for m in matched if 'oxford_cefr_a1' in m])/len(matched)
    e['cefr_A2'] = sum(
        [1 for m in matched if 'oxford_cefr_a2' in m])/len(matched)
    e['cefr_B1'] = sum(
        [1 for m in matched if 'oxford_cefr_b1' in m])/len(matched)
    e['cefr_B2'] = sum(
        [1 for m in matched if 'oxford_cefr_b2' in m])/len(matched)
    e['cefr_C1'] = sum(
        [1 for m in matched if 'oxford_cefr_c1' in m])/len(matched)
    return e


def add_lemmatize(e):
    e['lemmatized'] = [lemmatize(w)[0] for w in e['text'].split()]
    e['lemmatized_clean'] = [w for w in e['lemmatized']
                             if (w not in stop_words and not w.isnumeric())]
    return e


if __name__ == '__main__':

    df = pd.read_csv('cefr_leveled_texts.csv')
    ds = datasets.Dataset.from_pandas(df)
    ds.map(add_lemmatize, num_proc=8).filter(lambda e: len(e['lemmatized_clean']) > 0).map(
        add_word_freq, num_proc=8).map(add_oxford, num_proc=8).to_csv('analyzed.csv')
