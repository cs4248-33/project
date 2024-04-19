import uuid
import spacy
from typing import List, Dict, Set
from nltk import pos_tag
from util import chunk_list
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from multiprocessing import cpu_count, Manager
from multiprocessing.pool import Pool

def process_batch(
    inputs: List[str], 
    ooc_syns_dict: Dict[str, Set[str]], 
    n_generate: int
) -> List[str]:
    id = uuid.uuid4()
    nlp = spacy.load('en_core_web_md')

    augmented_sentences = []

    for input_sentence in inputs:
        input_sentence = input_sentence.lower()
        words = word_tokenize(input_sentence)
        tagged_words = pos_tag(words)

        for ooc_word, synonyms in ooc_syns_dict.items():
            # process the ooc word to get its text and POS
            ooc_pos_tag = nlp(ooc_word)[0].tag_

            words_to_replace = set([word for word, pos in tagged_words if pos == ooc_pos_tag and word in synonyms])

            for syn in words_to_replace:
                augmented_sentences.append(input_sentence.replace(syn, ooc_word, 1))
                
                count = len(augmented_sentences)
                # max of n_generate sentences will be created
                if count >= n_generate: 
                    return augmented_sentences
                elif count == 1 or count % 100 == 0:
                    print(f"{id} {count}: {input_sentence}")
                    print(f"Replaced {syn} <-> {ooc_word}")
                    
                    
    return augmented_sentences

def synonym_substitution(
    inputs: List[str], 
    ooc_words: List[str], 
    n_generate: int=20000
) -> List[str]:
    num_threads = min(10, cpu_count())
    input_batches = chunk_list(inputs)
    n_generate_per_batch = n_generate // num_threads

    print("num_threads", num_threads)
    print("n_generate_per_batch", n_generate_per_batch)

    def flatten(xss): return [x for xs in xss for x in xs]

    syns_dict = { 
        word.lower() : set(flatten(wordnet.synonyms(word.lower()))) for word in ooc_words 
    }

    with Manager() as manager:
        augmented_sentences = manager.list()

        with Pool() as pool:
            args = [(batch, syns_dict, n_generate_per_batch) for batch in input_batches]
            for result in pool.starmap_async(process_batch, args).get():
                augmented_sentences.extend(result)

        return list(augmented_sentences)
    