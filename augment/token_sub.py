import spacy
import random
from typing import List
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

# used when checking word in sentence is synonymous with the ooc word
def are_synonyms(word1, word2, pos, threshold):
    for syn1 in wordnet.synsets(word1):
        for syn2 in wordnet.synsets(word2):
            if syn1.pos() in pos and syn2.pos() in pos:
                similarity = syn1.wup_similarity(syn2)
                if similarity is not None and similarity >= threshold:
                    return True
    return False


# for token substitution, returns only the list of augmented sentences
# assume inputs is a list of English sentences
# assume ooc_words is a list of words
# K is the maximum number of sentences that each ooc word will augment.
# i.e. if K=50 and we have 50 ooc words, we can augment up to 50x50 sentences
def token_substitution(
    inputs: List[str], 
    ooc_words: List[str], 
    K: int=50,
    synonym_threshold: float=0.6
) -> List[str]:
    nlp = spacy.load('en_core_web_md')
    
    augmented_sentences = []

    for ooc_word in ooc_words:
        selected_indices = set()

        # process the ooc word to get its text and POS
        ooc_word_struct = nlp(ooc_word)[0]

        # Randomly select max K sentences from inputs to augment
        # for each ooc word, max of K sentences will be augmented
        # some sentences may not be augmented if there are no synonyms present
        while len(selected_indices) < K:
            random_index = random.randint(0, len(inputs) - 1)
            if random_index in selected_indices:
                print("[token_sub_aug]: skipping due to index collision")
                continue

            input_sentence = inputs[random_index]

            # Tokenize input sentence
            words = word_tokenize(input_sentence)

            # Perform POS tagging
            tagged_words = pos_tag(words)

            # Find words with the same POS tag as ooc_word
            same_pos_words = [word for word, pos in tagged_words if pos == ooc_word_struct.tag_]

            # Perform token substitution
            augmented_sentence = input_sentence
            is_augmented = False
            for word in same_pos_words:
                if are_synonyms(word, ooc_word_struct.text, get_wn_pos(ooc_word_struct.tag_), synonym_threshold):
                    augmented_sentence = augmented_sentence.replace(word, ooc_word_struct.text)
                    is_augmented = True

            if is_augmented:
                augmented_sentences.append(augmented_sentence)
            
            selected_indices.add(random_index)

    return augmented_sentences