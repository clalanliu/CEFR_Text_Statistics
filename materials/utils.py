import importlib

import nltk
from nltk.stem.wordnet import WordNetLemmatizer


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
            module = importlib.import_module(
                f"ai_knowitall.materials.{language}.vocab.{module_name}")
            modules[module_name] = module
        except ModuleNotFoundError:
            print(f"Module {module_name} not found.")
    return modules


def match_word_with_banks(word, lemmatized=False, language='english', module_names=None):
    if language == 'english':
        if module_names is None:
            module_names = ['ielts', 'toefl', 'toeic', 'gre',
                            'business', 'academic', 'tw_senior_high', 'oxford']

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
        module = importlib.import_module(
            f"ai_knowitall.materials.{language}.vocab")

    if word in module.word_freq:
        return module.word_freq[word]
    elif lemmatize(word)[0] in module.word_freq:
        return module.word_freq[lemmatize(word)[0]]
    return 0


if __name__ == '__main__':
    print(match_word_with_banks('abnormal'))
    print(match_word_with_banks('yield'))
    print(match_word_with_banks('inefficient'))
