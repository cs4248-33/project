import spacy
from typing import List
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

def synonym_substitution(
    inputs: List[str], 
    ooc_words: List[str], 
    n_generate: int = 20000
) -> List[str]:
    def flatten(xss): return [x for xs in xss for x in xs]

    nlp = spacy.load('en_core_web_md')

    syns_dict = { 
        word.lower() : flatten(wordnet.synonyms(word.lower())) for word in ooc_words 
    }

    augmented_sentences = []

    for input_sentence in inputs:
        input_sentence = input_sentence.lower()
        words = word_tokenize(input_sentence)
        tagged_words = pos_tag(words)

        for ooc_word, synonyms in syns_dict.items():
            # process the ooc word to get its text and POS
            ooc_pos_tag = nlp(ooc_word)[0].tag_

            words_to_replace = set([word for word, pos in tagged_words if pos == ooc_pos_tag and word in synonyms])

            for syn in words_to_replace:
                augmented_sentences.append(input_sentence.replace(syn, ooc_word))
                
                count = len(augmented_sentences)
                # max of n_generate sentences will be created
                if count >= n_generate: 
                    return augmented_sentences
                elif count % 100 == 0:
                    print(f"{count}: input_sentence")
                    print(f"Replaced {syn} <-> {ooc_word}")
                    
                    
    return augmented_sentences