import spacy
from typing import List
from nltk import pos_tag
from nltk.tokenize import word_tokenize

# for token substitution, returns only the list of augmented sentences
# assume inputs is a list of English sentences
# assume ooc_words is a list of words
def token_substitution(
    inputs: List[str], 
    ooc_words: List[str], 
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

            # Since we use a set (which is unordered), this will replace a random word
            augmented_sentence = input_sentence.replace(next(iter(same_pos_words)), ooc_word, 1)
            augmented_sentences.append(augmented_sentence)

    return augmented_sentences