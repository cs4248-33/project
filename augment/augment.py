import os
import json
import argparse
from token_syn import synonym_substitution
from token_sub import token_substitution
from const_sub import constituency_sub_augmentation
from parse_tree_sub import parse_tree_sub_augmentation
from datasets import load_dataset
from typing import List, Optional

def main(
    input_json_path: str, 
    ooc_words_txt_path: str, 
    output_json_path: Optional[str],
    augment_func
):
    """"
    augment_func should accept two lists of strings:
        1. input_sentences: List[str]
        2. ooc_words: List[str]
    And returns another list of strings which represent the augmented sentences
    """
    with open(ooc_words_txt_path) as f:
        ooc_words = [line.lower().strip() for line in f.readlines()]

    valid_extensions = ["json", "jsonl"]
    if input_json_path is not None:
        extension = input_json_path.split(".")[-1]
        assert extension in valid_extensions, "`train_file` should be a jsonlines file."

    raw_dataset = load_dataset(
        "json",
        data_files=input_json_path,
        cache_dir=None,
        token=None,
    )["train"].shuffle(seed=1)
    
    inputs = [pair["translation"]["en"] for pair in raw_dataset]

    augmented_sentences = augment_func(inputs, ooc_words)
    
    if output_json_path is not None:
        with open(output_json_path, mode="w") as f:
            for sentence in augmented_sentences:
                pair = {}
                pair["translation"] = {}
                pair["translation"]["en"] = sentence
                pair["translation"]["zh"] = ""
                json.dump(pair, f)
                f.write("\n")
 
    return augmented_sentences

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json_path", 
        help="Path to JSONLINES file containing the training data to be augmented",
        type=str
    )
    parser.add_argument(
        "--ooc_words_txt_path", 
        help="Path to .txt file containing the list of OOC words, separated by newlines",
        type=str
    )
    parser.add_argument(
        "--output_json_path", 
        help="Path to JSONLINES file which will be overwritten with the augmented data",
        type=str
    )
    parser.add_argument(
        "--strategy", 
        help="Augmentation strategy to apply. Valid arguments include TOK, TOKSYN, PARSE, CONST",
        type=str
    )
    parser.add_argument(
        "--n_generate", 
        help="The maximum number of augmented sentences to generate",
        type=int
    )

    args = parser.parse_args()

    assert args.input_json_path is not None
    assert args.ooc_words_txt_path is not None
    assert args.output_json_path is not None
    assert args.strategy is not None 
    assert args.strategy in ["TOK", "TOKSYN", "PARSE", "CONST"]
    assert args.n_generate is not None

    input_json_path = args.input_json_path
    ooc_words_txt_path = args.ooc_words_txt_path
    output_json_path = args.output_json_path
    strategy = args.strategy
    n_generate = args.n_generate

    if strategy == "TOK":
        augment_func = lambda inputs, ooc_words: token_substitution(inputs, ooc_words, n_generate)
    elif strategy == "TOKSYN":
        augment_func = lambda inputs, ooc_words: synonym_substitution(inputs, ooc_words, n_generate)
    elif strategy == "CONST":
        augment_func = lambda inputs, ooc_words: constituency_sub_augmentation(inputs, ooc_words, n_generate)
    elif strategy == "PARSE":
        augment_func = lambda inputs, ooc_words: parse_tree_sub_augmentation(inputs, ooc_words, n_generate)

    print(f"Generating up to {n_generate} sentences with {strategy}")

    augmented_sentences = main(
        input_json_path=input_json_path,
        ooc_words_txt_path=ooc_words_txt_path, 
        output_json_path=output_json_path,
        augment_func=augment_func
    )