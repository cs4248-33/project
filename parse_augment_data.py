import os
import random
import json
import nltk
from datasets import load_dataset
from deep_translator import GoogleTranslator
from nltk.data import find
from bllipparser import RerankingParser

def get_augment_sentence(sentence, ooc_set, parser):
    best_sentence = None
    best_score = 0
    words = sentence.split()

    for pos in range(len(words)):
        for ooc_word in ooc_set:
            temp_words = words.copy()
            temp_words[pos] = ooc_word
            sentence_with_word_replaced_with_ooc_word = " ".join(temp_words)
            nbest_list = parser.parse(sentence_with_word_replaced_with_ooc_word)

            if len(nbest_list) > 0:
                # Get the best parse according to the reranker
                best_parse = nbest_list.get_reranker_best()
                reranker_score = abs(best_parse.reranker_score)
                # Update the best sentence if this parse is better than the current best
                if reranker_score > best_score:
                    best_score = reranker_score
                    best_sentence = sentence_with_word_replaced_with_ooc_word

    # Translate the best sentence
    if best_sentence:
        translated_sentence = GoogleTranslator(source='en', target='zh-CN').translate(best_sentence)
    else:
        translated_sentence = None

    return best_sentence, translated_sentence


def augment_dataset_and_save(ooc_set, num_sentences_to_augment=100):
    data_dir = "./data"
    splits = ["train", "validation"]
    dataset = load_dataset("iwslt2017", "iwslt2017-en-zh", data_dir=data_dir, cache_dir=data_dir)

    # Initialize the translator
    model_dir = find('models/bllip_wsj_no_aux').path
    parser = RerankingParser.from_unified_model_dir(model_dir)

    for split in splits:
        all_sentences = []

        # Randomly choose sentence to do OOC word replacement
        indices = list(range(len(dataset[split])))
        random_indices = random.sample(indices, min(num_sentences_to_augment, len(indices)))

        for i in random_indices:
            example = dataset[split][i]
            source_sentence = example['translation']['en']
            target_sentence = example['translation']['zh']

            # # Add original sentence pair
            # all_sentences.append({
            #     'translation': {
            #         'en': source_sentence,
            #         'zh': target_sentence
            #     }
            # })
            # Replace with OOC word and add new sentence pair
            # if i in random_indices:
            aug_source, aug_target = get_augment_sentence(source_sentence, ooc_set, parser)
            if aug_source and aug_target:
                all_sentences.append({
                    'translation': {
                        'en': aug_source,
                        'zh': aug_target
                    }
                })

        # Save both original and augmented sentences to JSON
        path = os.path.join(data_dir, f"{split}_parse_augmented.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(all_sentences, f, ensure_ascii=False)

        print(f"Augmented sentences saved in {path}")

if __name__ == "__main__":
    nltk.download('bllip_wsj_no_aux')
    ooc_set = {'cat', 'dog'}
    augment_dataset_and_save(ooc_set, num_sentences_to_augment=100)
