import joblib
import os
from ai_knowitall.materials.utils import get_word_freq, match_word_with_banks
import numpy as np
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

model_folder = os.path.dirname(__file__)
# Load the model from the file
classifier = joblib.load(os.path.join(model_folder, 'cefr_classifier.joblib'))

# Load the label encoder from the file
label_encoder = joblib.load(os.path.join(model_folder, 'cefr_label_encoder.joblib'))


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
    e['word_freq_75'] = np.percentile(freqs, 75)
    e['word_freq_25'] = np.percentile(freqs, 25)
    e['word_freq_mean'] = np.mean(freqs)
    return e


def add_oxford(e):
    matched = [match_word_with_banks(w) for w in e['lemmatized_clean']]
    e['cefr_A1'] = sum([1 for m in matched if 'oxford_cefr_a1' in m])/len(matched)
    e['cefr_A2'] = sum([1 for m in matched if 'oxford_cefr_a2' in m])/len(matched)
    e['cefr_B1'] = sum([1 for m in matched if 'oxford_cefr_b1' in m])/len(matched)
    e['cefr_B2'] = sum([1 for m in matched if 'oxford_cefr_b2' in m])/len(matched)
    e['cefr_C1'] = sum([1 for m in matched if 'oxford_cefr_c1' in m])/len(matched)
    return e


def classify_text(text):

    d = {}
    d['lemmatized_clean'] = [w for w in [
        lemmatize(w)[0] for w in text.split()] if (w not in stop_words and not w.isnumeric())]

    d = add_oxford(d)
    d = add_word_freq(d)
    label = classifier.predict([[d['word_freq_mean'], d['word_freq_75'], d['word_freq_25'], d['cefr_A1'],
                               d['cefr_A2'], d['cefr_B1'], d['cefr_B2'], d['cefr_C1']]])
    return label_encoder.inverse_transform(label)[0]


if __name__ == '__main__':
    print(classify_text('I am the best'))
    print(classify_text("Hi! I've been meaning to write for ages and finally today I'm actually doing something about it. Not that I'm trying to make excuses for myself, it's been really hard to sit down and write, as I've been moving around so much. Since we last saw each other I've unpacked my bags in four different cities. This job has turned out to be more of a whirlwind than I expected, but it's all good! I went from London to Prague to set up a new regional office there. You know I'd always wanted to go, but maybe I was imagining Prague in spring when I used to talk about that. Winter was really hard, with minus 15 degrees in the mornings and dark really early in the evening. But at least it was blue skies and white snow and not days on end of grey skies and rain, like at home. It's tough being away from home over Christmas, though, and Skype on Christmas Day wasn't really the same as being with everyone. From there I was on another three-month mission to oversee the set-up of the office in New York. Loved, loved, loved New York! It's like being in one big TV show, as everywhere looks just a little bit familiar. I did every tourist thing you can think of when I wasn't working, and must have spent most of my salary on eating out. It was really hard to leave for the next job, especially as I kind of met someone (!) More about Michael later ... So then I was posted to LA, which felt like a whole other country compared with the East Coast. I could definitely get used to that kind of outdoor, beach lifestyle, but I didn't spend as much time getting to know California as I could have because I was flying back to see Michael every other weekend. He came to see me when he could, but his job means he's often working at weekends, so he couldn't make the flight very often. Those three months flew by and then I was off again, to Frankfurt, which is where I am now. And … so is Michael! He got a month off work and we're trying to work out how we can be in the same place at the same time for a while. We figure the first step in that direction is getting married, which is also why I wanted to write – I can't get married without my oldest friend there! The wedding's going to be at home in London in September and I hope you can come! Anyway, tell me all your news and I promise not to leave it so long this time! Lots of love, Kath"))
