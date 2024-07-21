import importlib
from functools import cmp_to_key
import pandas as pd
import nltk
from nltk.stem.wordnet import WordNetLemmatizer

from ai_knowitall.materials.english.vocab.cefr import vocabs as english_vocabs

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

def get_module(module_names, language):
    modules = {}
    for module_name in module_names:
        try:
            module = importlib.import_module(f"ai_knowitall.materials.{language}.vocab.{module_name}")
            modules[module_name] = module
        except ModuleNotFoundError:
            print(f"Module {module_name} not found.")
    return modules


def match_word_with_banks(word, lemmatized=False, language='english', module_names=None):
    if language == 'english':
        if module_names is None:
            module_names = ['ielts', 'toefl', 'toeic', 'gre', 'business',
                            'academic', 'tw_senior_high', 'oxford', 'cefr']

    if not lemmatized:
        word = lemmatize(word)[0]
    modules = get_module(module_names, language=language)
    matched_set = []
    for module_name in modules:
        if hasattr(modules[module_name], 'vocabs'):
            for k, v in modules[module_name].vocabs.items():
                if word in v:
                    matched_set.append(k)

    return matched_set


def get_word_freq(word, language='english'):
    if language == 'english':
        module = importlib.import_module(f"ai_knowitall.materials.{language}.vocab")

    if word in module.word_freq:
        return module.word_freq[word]
    elif lemmatize(word)[0] in module.word_freq:
        return module.word_freq[lemmatize(word)[0]]
    return 0


def sort_word_difficulties(words, key=lambda x: x, language='english', return_difficulties=False):
    def word_difficulty(word):
        word_ = lemmatize(word)[0]
        freq = get_word_freq(word_, language)
        level = next((level_ for level_, vocab in english_vocabs.items() if word_ in vocab), None)
        return freq, level

    class Difficulty:
        def __init__(self, item):
            self.item = item
            if isinstance(item, dict):
                self.word = key(item)
            else:
                self.word = item
            self.freq, self.level = word_difficulty(self.word)

        def __lt__(self, other):
            if isinstance(self.level, str) and isinstance(other.level, str):
                return self.level < other.level
            return self.freq > other.freq

    words_sorted = sorted(words, key=Difficulty)

    if return_difficulties:
        words_info = [{'word': key(w), "difficulty": word_difficulty(key(w))} for w in words_sorted]
        return words_sorted, words_info
    else:
        return words_sorted

if __name__ == '__main__':
    print(match_word_with_banks('abnormal'))
    print(match_word_with_banks('yield'))
    print(match_word_with_banks('inefficient'))

    print(sort_word_difficulties(words=['upgrade', 'upon', 'upstairs']))
    print(sort_word_difficulties(words=['rescue', 'eye', 'smother']))

    content = """
Rescued this friendly stray and named her PINKY.  She was starving when I got her. I took her to the vet and she wasn't fixed nor microchipped and she has and eye issue.       
I got her sprayed, vaccinated, and fully vetted. The next stop is taking her to the ophthalmologist for her eye.
"""
    print(sort_word_difficulties(words=content.split()))
