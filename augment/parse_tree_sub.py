import uuid
import random
import nltk
import spacy
from typing import List
from nltk import pos_tag
from nltk.data import find
from nltk.tokenize import word_tokenize
from bllipparser import RerankingParser
from multiprocessing import cpu_count, Manager
from multiprocessing.pool import Pool

nltk.download('bllip_wsj_no_aux')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def get_augment_sentence(words, ooc_word, parser, nlp):
    # Just get the new english sentence first, get translation using baseline separately then combine
    best_sentence = None
    best_score = 0

    # Perform POS tagging
    # Process the ooc word to get its text and POS
    ooc_word_struct = nlp(ooc_word)[0]
    tagged_words = pos_tag(words)
    same_pos_words_indices = [i for i, (word, pos) in enumerate(tagged_words) if pos == ooc_word_struct.tag_]

    for pos in range(len(same_pos_words_indices)):
        original_word = words[pos]
        words[pos] = ooc_word  # Modify the list in place
        nbest_list = parser.parse(words, rerank=False)

        if len(nbest_list) > 0:
            # Get the best parse according to the reranker
            best_parse = nbest_list.get_parser_best()
            parser_score = abs(best_parse.parser_score)
            # Update the best sentence if this parse is better than the current best
            if parser_score > best_score:
                best_score = parser_score
                best_sentence = " ".join(words)

        words[pos] = original_word  # Revert the modification

    return best_sentence

def process_batch(
    inputs: List[str], 
    ooc_words: List[str], 
    n_generate: int=20000
) -> List[str]:
    id = uuid.uuid4()

    # Initialize the parser
    model_dir = find('models/bllip_wsj_no_aux').path
    parser = RerankingParser.from_unified_model_dir(model_dir)
    nlp = spacy.load('en_core_web_md')

    augmented_sentences = []
    ooc_words_count = len(ooc_words)
    inputs_count = len(inputs)

    while len(augmented_sentences) < n_generate:
        for ooc_word in ooc_words:
            input_sentence_index = random.randint(0, inputs_count-1)
            input_sentence = inputs[input_sentence_index]

            words = word_tokenize(input_sentence)
            augmented = get_augment_sentence(words, ooc_word, parser, nlp)

            if augmented:
                augmented_sentences.append(augmented)
                if len(augmented_sentences) % 100 == 0:
                    print(id, "AUGMENTED", len(augmented_sentences))
                if len(augmented_sentences) >= n_generate:
                    return augmented_sentences

    return augmented_sentences

def parse_tree_sub_augmentation(
    inputs: List[str], 
    ooc_words: List[str], 
    n_generate: int=20000
) -> List[str]:
    batch_size = len(inputs) // 10
    input_batches = [inputs[i:i+batch_size] for i in range(0, len(inputs), batch_size)]
    num_threads = len(input_batches)
    n_generate_per_batch = n_generate // num_threads

    print("batch_size", batch_size)
    print("num_threads", num_threads)
    print("n_generate_per_batch", n_generate_per_batch)
    with Manager() as manager:
        augmented_sentences = manager.list()

        with Pool() as pool:
            args = [(batch, ooc_words, n_generate_per_batch) for batch in input_batches]
            for result in pool.starmap_async(process_batch, args).get():
                print(result)
                augmented_sentences.extend(result)

        return list(augmented_sentences)