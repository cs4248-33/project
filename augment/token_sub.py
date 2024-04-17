import math
import spacy
import random
from typing import List, Optional
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

# simple conversion to WordNet POS tags
def get_wn_pos(pos):
    first_char = pos[0].lower()
    if first_char == 'n':
        return 'n'  # Noun
    elif first_char == 'v':
        return 'v'  # Verb
    elif first_char == 'j':
        return 'as'  # Adjective
    elif first_char == 'r':
        return 'r'  # Adverb
    else:
        return None

# unused, just for testing
def get_synonyms(word, pos, threshold):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.name() != word and lemma.synset().pos() == pos:
                similarity = syn.wup_similarity(lemma.synset())
                if similarity is not None and similarity >= threshold:
                    synonyms.append(lemma.name())
    return synonyms

def get_similarity_score(word1: str, word2: str, pos: str) -> Optional[float]:
    for syn1 in wordnet.synsets(word1):
        for syn2 in wordnet.synsets(word2):
            if syn1.pos() in pos and syn2.pos() in pos:
                return syn1.wup_similarity(syn2)
    return None

# used when checking word in sentence is synonymous with the ooc word
def are_synonyms(word1: str, word2: str, pos: str, threshold: float) -> bool:
    similarity = get_similarity_score(word1, word2, pos)
    return similarity is not None and similarity >= threshold

# for token substitution, returns only the list of augmented sentences
# assume inputs is a list of English sentences
# assume ooc_words is a list of words
def token_substitution(
    inputs: List[str], 
    ooc_words: List[str], 
    synonym_threshold: float=0.6,
    n_generate: int = 20000
) -> List[str]:
    nlp = spacy.load('en_core_web_md')
    
    augmented_sentences = []

    for input_sentence in inputs:
        # max of n_generate sentences will be created
        if len(augmented_sentences) >= n_generate: 
            return augmented_sentences
        
        for ooc_word in ooc_words:
            # process the ooc word to get its text and POS
            ooc_word_struct = nlp(ooc_word)[0]
            ooc_word = ooc_word_struct.text
            ooc_pos_tag = ooc_word_struct.tag_

            # Tokenize input sentence
            words = word_tokenize(input_sentence)

            # Perform POS tagging
            tagged_words = pos_tag(words)

            # Find words with the same POS tag as ooc_word
            same_pos_words = set([word for word, pos in tagged_words if pos == ooc_word_struct.tag_])

            if len(same_pos_words) <= 0: continue

            if synonym_threshold is None:
                # Since we use a set (which is unordered), this will replace a random word
                augmented_sentence = input_sentence.replace(same_pos_words[0], ooc_word, 1)
                augmented_sentences.append(augmented_sentence)
            else:
                best_similarity_score = -math.inf
                best_replacement_word = None

                # Replace the word with the highest similarity score that shares
                # its POS tag with the ooc_word. The similarity score must be higher than synonym_threshold to be considered.
                for word in same_pos_words:
                    similarity_score = get_similarity_score(word, ooc_word, ooc_pos_tag)
                    if similarity_score < synonym_threshold: 
                        continue
                    elif similarity_score > best_similarity_score:
                        best_similarity_score = similarity_score
                        best_replacement_word = word

                if best_replacement_word is not None:
                    augmented_sentence = input_sentence.replace(word, ooc_word)
                    augmented_sentences.append(augmented_sentence)

    return augmented_sentences