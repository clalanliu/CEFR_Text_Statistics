"""
L_algorithm.py Idiomatic Expression Matcher

This script is designed to find idiomatic expressions within English sentences by comparing them
to a dictionary of phrases stored in 'phrases.json'.
It consists of two main functions, 'get_potential_matches()' and 'look_closer()', 
which work together to identify potential dictionary entries that match a given English sentence
and then filter and rank those potential matches to find idiomatic expression matches.

'get_potential_matches()' Function:
-------------------------------------
This function aims to efficiently narrow down the search for potential dictionary entries that match
an input English sentence. It iterates through all entries in 'phrases.json' and checks if the 
sentence contains all the "constant" elements present in a dictionary entry. 
Various strategies are employed, including considering multi-word constants, word forms, 
and lemmatization, to enhance matching capabilities.

Args:
    sentence (str): The English sentence to search for potential matches within the 
    dictionary entries.

Returns:
    list: A list of indices corresponding to potential matching entries in 'phrases.json'. 
    This function significantly reduces the number of entries to be analyzed in the subsequent 
    matching process.

'look_closer()' Function:
--------------------------
This function refines the list of potential matches obtained from 'get_potential_matches()' 
to identify idiomatic expressions in the input sentence. 
It considers exact word matches, lemmatized forms, and patterns to determine the presence of 
idiomatic expressions. 
The resulting matches are filtered and sorted based on the number of matching words and word span.

Args:
    potential_matches (list): A list of potential matches returned by the 
    'get_potential_matches()' function.
    sentence (str): The English sentence to search for idiomatic expressions.

Returns:
    list: A list of filtered and sorted idiomatic expression matches. 
    Each item in the list consists of the dictionary entry, the number of matching words, 
    and the matched word span.

Note:
The script includes other auxiliary functions such as 'get_match_span()' and 'is_it_sorted()'
for various calculations and checks.

Example Usage:
--------------
Given an input sentence and dictionary:
```python
sentence = "The children were accustomed to eating late in the evening."
potential_matches = get_potential_matches(sentence)
idiomatic_matches = look_closer(potential_matches, sentence)
print(idiomatic_matches)
"""

from collections import Counter
import hashlib
import itertools
import json
import os
import pprint
import re
import uuid
from warnings import warn
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('popular')
# Determine the package-relative path to phrases.json
package_dir = os.path.dirname(__file__)

# manually point nltk to my top level nltk_data folder that includes wordnet
# https://www.nltk.org/data.html
nltk_data_dir = os.path.join(package_dir, "nltk_data")
nltk.data.path.append(nltk_data_dir)


def precompile_regexes(dictionary, regexes=None):
    '''
    regexes: dict of re.compile(...). This function will check `dictionary` and add keys and values to extend regexes. If not passed, will be created as `{}`.
    '''
    if regexes is None:
        regexes = {}
    for entry in dictionary:
        constant_elements = [(a, r, wf) for a, r, wf in zip(
            entry["alt"], entry["runs"], entry["word_forms"]) if wf != "NA"]
        for a, r, wf in constant_elements:
            if len(r.split()) == 1 and '|'.join(wf[0]) not in regexes:  # a single word constant
                regexes['|'.join(wf[0])] = re.compile(
                    rf"(?:^|\W|\b)({'|'.join(wf[0])})(?:\W|\b|$)", re.IGNORECASE
                )
            elif len(r.split()) > 1:
                for c, w in enumerate(r.split()):
                    escaped_word_forms = [re.escape(word) for word in wf[c]]
                    joined = '|'.join(escaped_word_forms)
                    if joined not in regexes:
                        regexes[joined] = re.compile(
                            rf"(?:^|\W|\b)({joined})(?:\W|\b|$)",
                            re.IGNORECASE,
                        )
    return regexes


def extend_dictionary(dictionary, extension_data, regexes=None):
    """
    This function extend the dictionary with extension_data and calls `precompile_regexes` and return extended `dictionary` and `regexes`.
    """
    for d in extension_data:
        d['id'] = uuid.UUID(bytes=hashlib.sha256(d["phrase"].encode("utf-8")).digest()[:16]).hex
        dictionary.append(d.copy())
    regexes = precompile_regexes(dictionary, regexes)
    return dictionary, regexes



def get_match_span(tuple_list):
    flat_list = [item for pair in tuple_list for item in pair]

    return flat_list[0], flat_list[-1]


def is_it_sorted(tuple_list):
    flat_list = [item for pair in tuple_list for item in pair]

    if sorted(flat_list) == flat_list:
        return True

    return False

def is_period_between_words(words, sentence):
    # Create patterns to match consecutive pairs of words without a period between them
    for i in range(len(words) - 1):
        pattern_no_period = re.escape(words[i]) + r'[^.]*' + re.escape(words[i + 1])
        # Check if there is any match for these patterns
        if re.search(pattern_no_period, sentence):
            return False

    # Create patterns to match each word followed by any number of characters and a period
    period_patterns = [re.escape(word) + r'[^.]*\.' for word in words]

    # Check if there are matches for these patterns
    for pattern in period_patterns:
        if not re.findall(pattern, sentence):
            return False

    return True


def longest(l):
    if not isinstance(l, list):
        return 0
    return max(
        [len(l)]
        + [len(subl) for subl in l if isinstance(subl, list)]
        + [longest(subl) for subl in l]
    )


def max_words_between(sentence, words, spans):
    # Initialize max_words to zero
    max_words = 0

    # Iterate through pairs of adjacent words
    for i in range(len(words) - 1):
        word1 = words[i]
        word2 = words[i + 1]

        # Get the span of each word
        span1 = spans[i]
        span2 = spans[i + 1]

        # Calculate the number of words between the two spans
        words_between = len(sentence[span1[1] + 1: span2[0]].split())

        # Update max_words if necessary
        if words_between > max_words:
            max_words = words_between

    return max_words


def generate_combinations_with_constants(input_list):
    # Extract variant items and constant items
    variant_items = [item for item in input_list if isinstance(item, list)]
    constant_items = [item for item in input_list if not isinstance(item, list)]

    # Generate all possible combinations for variant items
    variant_combinations = itertools.product(*variant_items)

    # Combine variant combinations with constant items
    result = []
    for variant_combination in variant_combinations:
        combined = []
        variant_index = 0
        for item in input_list:
            if isinstance(item, list):
                combined.append(variant_combination[variant_index])
                variant_index += 1
            else:
                combined.append(item)
        result.append(tuple(combined))

    return result


def get_potential_matches(sentence, sentence_lemma, dictionary, regexes):
    potential_matches = []

    for entry in dictionary:
        constant_elements = [(a, r, wf) for a, r, wf in zip(
            entry["alt"], entry["runs"], entry["word_forms"]) if a == "constant"]
        constant_count = len(constant_elements)
        constant_match_count = 0

        for a, r, wf in constant_elements:
            if r in sentence:
                constant_match_count += 1

            elif len(r.split()) == 1:  # a single word constant
                # let's see if any of the word forms of the constant exist in the sentence
                p = regexes['|'.join(wf[0])]

                m = p.search(sentence)

                if m is not None:
                    constant_match_count += 1
                else:
                    # does any of the words from the word form list exist
                    # in the lemmatized sentence?
                    if any(w in wf[0] for w in sentence_lemma):
                        constant_match_count += 1

            elif len(r.split()) > 1:  # a multiple word constant
                # break down the constant to words and see if all those words
                # (or their forms) exist in the sentence

                word_match_count = 0

                for c, w in enumerate(r.split()):
                    if w in sentence:
                        word_match_count += 1
                    else:
                        # let's see if any of the word forms exist in the sentence

                        # Escape special characters in word forms to prevent regex errors
                        # When constructing regular expressions, certain characters such
                        # as '+', '*', '^', '$', etc., have special meanings.
                        # If these characters appear in the word forms and are not properly
                        # escaped, they can lead to errors during regex pattern compilation.
                        escaped_word_forms = [re.escape(word) for word in wf[c]]
                        p = regexes['|'.join(escaped_word_forms)]

                        m = p.search(sentence)

                        if m is not None:
                            word_match_count += 1
                        else:
                            if any(w in wf[c] for w in sentence_lemma):
                                word_match_count += 1

                if word_match_count == len(r.split()):
                    constant_match_count += 1

        if constant_count == constant_match_count:
            potential_matches.append(entry)

    return potential_matches


def look_closer(potential_matches, sentence, sentence_lemma, dictionary, regexes):
    """
    Filters potential matches and provides a list of idiomatic expression matches
    in a given English sentence.

    This function refines a list of potential matches obtained from the `get_potential_matches()`
    function by examining various factors, including exact word matches, lemmatized forms,
    and patterns, to determine the presence of idiomatic expressions in the input sentence.
    The workflow consists of the following steps:

    1. It considers all "article," "verb," "o-constant," and "constant" elements in each potential
       match and searches for their exact word, lemmatized form, or any form in the input sentence.
    2. It generates patterns from the findings and compares these patterns to those in the
       dictionary entry.
    3. It checks that all word matches in the sentence appear in the same order as in the
       dictionary entry.
    4. It ensures that matching words are not too far apart in the sentence, with a maximum allowed
       distance of 3 words between matched words to avoid incorrect matches in long sentences.
    5. The function returns all matches sorted in descending order, with the entry that has the
       highest number of matching words listed first.

    Args:
    potential_matches (list): A list of potential matches returned by get_potential_matches().
    sentence (str): The English sentence to search for idiomatic expressions.

    Returns:
    list: A list of filtered and sorted idiomatic expression matches, with each item consisting
    of the dictionary entry, the number of matching words, and the matched word span.

    Example:
    Given a list of potential matches and an input sentence:
    potential_matches = [0, 2, 4]  # Indices of potential matches in 'phrases.json'
    sentence = "The children were accustomed to eating late in the evening."

    The function returns a list of filtered matches, such as:
    [
        (matching_entry_1, 3, [(13, 31)]),
        (matching_entry_2, 1, [(32, 38)]),
        # ...
    ]
    """
    # Notes:
    # [refined_matches] always has fewer matches than potential_matches,
    # but the problem is that there are lots of entries in phrases.json that can
    # be triggered by a single word in a sentence - for example: the word "want" will always
    # trigger range (84897,84900)
    # to narrow down results a bit more, i'm organizing matches in [refined_matches]
    # in descending order.
    # i'll put entries with the highest number of matches 1st
    # - hopefully i can always pick up a match from the 1st 3 suggestions

    matches = []
    sentence_lemma_string = " ".join(sentence_lemma)

    # loop through all potential_matches in [dictionary]
    for d in potential_matches:
        record = []
        ''' removed detection for checking whether , between words of idioms
        unvariable_words = " ".join([word for r, a in zip(d["runs"], d["alt"]) if a !=
                                    'variable' for word in lemmatize(r)]).split()
        if is_period_between_words(unvariable_words, sentence_lemma_string):
            continue
        '''
        for a, r, wf in zip(d["alt"], d["runs"], d["word_forms"]):
            if (
                a in ["article", "verb", "o-constant", "constant"]
                and len(r.split()) == 1
            ):
                p = regexes['|'.join(wf[0])]
                match = p.findall(
                    sentence
                )  # returns a list of all matches, or empty list in case of no match

                # Note: if '\W' matched in the regex pattern, the match would include a white space
                # - which will through the span off, in that case we need to capture group(1).
                # [span] first checks if m.groups() is not empty to ensure that the match has any
                # groups at all. Then it checks if m.group(1) exists before accessing it.
                # If both conditions are met, it captures m.span(1); otherwise, it captures m.span()
                span = [
                    m.span(1) if m.groups() and m.group(1) else m.span()
                    for m in p.finditer(sentence)
                ]  # returns a list of all m.span, or empty list in case of no match

                # update the record
                if len(match) > 1 and len(span) > 1:
                    record.append([(r, s) for m, s in zip(match, span)])
                elif match and span:
                    record.append((r, span[0]))
                else:
                    p = regexes['|'.join(wf[0])]
                    match = p.findall(sentence_lemma_string)
                    span = [
                        m.span(1) if m.groups() and m.group(1) else m.span()
                        for m in p.finditer(sentence_lemma_string)
                    ]

                    # update the record
                    if len(match) > 1 and len(span) > 1:
                        record.append([(r, s) for m, s in zip(match, span)])
                    elif match and span:
                        record.append((r, span[0]))

            elif (
                a in ["article", "verb", "o-constant", "constant"]
                and len(r.split()) > 1
            ):
                for c, w in enumerate(r.split()):
                    escaped_word_forms = [re.escape(word) for word in wf[c]]
                    p = re.compile(
                        rf"(?:^|\W|\b)({'|'.join(escaped_word_forms)})(?:\W|\b|$)",
                        re.IGNORECASE,
                    )

                    match = p.findall(sentence)
                    span = [
                        m.span(1) if m.groups() and m.group(1) else m.span()
                        for m in p.finditer(sentence)
                    ]

                    # add the values to the dictionary inside record
                    if len(match) > 1 and len(span) > 1:
                        record.append([(w, s) for m, s in zip(match, span)])
                    elif match and span:
                        record.append((w, span[0]))
                    else:
                        escaped_word_forms = [re.escape(word) for word in wf[c]]
                        p = re.compile(
                            rf"(?:^|\W|\b)({'|'.join(escaped_word_forms)})(?:\W|\b|$)",
                            re.IGNORECASE,
                        )

                        match = p.findall(sentence_lemma_string)
                        span = [
                            m.span(1) if m.groups() and m.group(1) else m.span()
                            for m in p.finditer(sentence_lemma_string)
                        ]

                        # update the record
                        if len(match) > 1 and len(span) > 1:
                            record.append([(w, s) for m, s in zip(match, span)])
                        elif match and span:
                            record.append((w, span[0]))

        combinations = generate_combinations_with_constants(record)

        for combo in combinations:
            words = [tupl[0] for tupl in combo]
            spans = [tupl[1] for tupl in combo]

            # lets check if we have a match
            if (
                words
                and spans
                and " ".join(words) in d["patterns"]
                and is_it_sorted(spans)
                and max_words_between(sentence, words, spans) <= 3
            ):
                matches.append((d, len(words), get_match_span(spans), spans))
                break

    # now, lets sort matches in descending order, highest number of matches 1st.
    # https://stackoverflow.com/questions/10695139/sort-a-list-of-tuples-by-2nd-item-integer-value
    sorted_refined_matches = sorted(matches, key=lambda x: x[1], reverse=True)

    return sorted_refined_matches


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


def refine_match_by_checking_word_existence_and_deduplicate(sentence: str, matches: list):
    def is_repetitive_phrase(counter, counter_list):
        for c in counter_list:
            if all(c[item] >= counter[item] for item in counter):
                return True
        return False

    refined_matches = []
    counter_list = []
    for m in matches:
        lemmatized_sentence = lemmatize(sentence[m[1][0]:m[1][1]])
        sentence_counter = Counter(lemmatized_sentence)
        phrase_list = [r for r, a in zip(m[0]['runs'], m[0]['alt']) if a == 'constant']
        phrase_counter = Counter(lemmatize(' '.join(phrase_list)))
        if is_repetitive_phrase(phrase_counter, counter_list):
            continue
        if all(sentence_counter[item] >= phrase_counter[item] for item in phrase_counter):
            refined_matches.append(m)
            counter_list.append(phrase_counter)
    return refined_matches


def split_into_sentences(text):
    # TODO: protect cases of abbrevation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences_indices = []
    search_start = 0
    for sentence in sentences:
        idx = text.find(sentence, search_start)
        if idx != -1:
            sentences_indices.append((sentence, idx))
            search_start = idx + len(sentence)
    return sentences_indices


def parse_definition_html(definition_html):
    definition_examples = re.split(r'<strong>\d+\. </strong>', definition_html)
    definitions = []
    for definition_example in definition_examples:
        if len(re.sub("<.+?>", "", definition_example)) > 0:
            definitions.append({
                "definition": re.sub("<.+?>", "", re.sub(r'<em>.+?</em>', "", definition_example.split('_')[0].strip())),
                'examples': [t.strip() for t in re.sub("<.+?>", "", definition_example).split('_')[1:]],
            })
    return definitions


def find_idioms_sentence(sentence, dictionary, regexes=None, limit=10, html=False, span=True, entry_range=True, entry_id=True, return_idx=True):
    if regexes is None:
        warn("You should pass regexes=precompile_regexes(dictionary) first to decrease the cost!")
        regexes = precompile_regexes(dictionary=dictionary, regexes={})

    sentence_lemma = lemmatize(sentence)
    potential_matches = get_potential_matches(
        sentence.lower(),
        set(sentence_lemma),
        dictionary,
        regexes
    )  # using set to increase speed
    lc = look_closer(potential_matches, sentence.lower(), sentence_lemma, dictionary, regexes)

    matches = list(
        k for k, _ in itertools.groupby([(item[0], item[2], item[3]) for item in lc[:limit]])
    )

    matches = refine_match_by_checking_word_existence_and_deduplicate(sentence, matches)
    # matches = label_occurence(sentence, matches)
    output = []
    for m in matches:
        tmp = {
            "runs": m[0]["runs"],
            "alt": m[0]["alt"]
        }
        if html:
            tmp.update(
                {
                    "phrase_html": m[0]["phrase_html"],
                    "definition_html": m[0]["definition_html"],
                }
            )
        else:
            tmp.update({
                "phrase": m[0]["phrase"],
                "definitions": parse_definition_html(m[0]["definition_html"]),
            })
        if span:
            tmp.update({"span": m[1]})
        if entry_range:
            tmp.update({"entry_range": m[0]["range"]})
        if entry_id:
            tmp.update({"entry_id": m[0]["id"]})
        if return_idx:
            tmp.update({"idx": m[2]})

        output.append(tmp)

    return output, sentence_lemma


def find_idioms(content, dictionary, regexes=None, limit=10, html=False, span=True, entry_range=True, entry_id=True, return_idx=True):
    sentences_indices = split_into_sentences(content)
    results = []
    for s, idx in sentences_indices:
        sentence_results, lemmatized_sentence = find_idioms_sentence(
            sentence=s, dictionary=dictionary, regexes=regexes, limit=limit, html=html, span=span, entry_range=entry_range, entry_id=entry_id, return_idx=return_idx)
        for sr in sentence_results:
            sr['span'] = [i + idx for i in sr['span']]
            sr['idx'] = (np.array(sr['idx']) + idx).tolist()
            results.append(sr)

    return results, lemmatize(content)

if __name__ == "__main__":
    pass
