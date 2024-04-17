import os
import json
from const_sub import constituency_sub_augmentation
from datasets import load_dataset, concatenate_datasets
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

def augment_training_data(
    train_json_path: str, 
    augment_json_path: str,
    translations_txt_path: str,
    concatenated_train_json_path: str,
):
    with open(translations_txt_path) as f:
        zh_translations = [line.lower().strip() for line in f.readlines()]

    train_dataset = load_dataset(
        "json",
        data_files=train_json_path,
        cache_dir=None,
        token=None,
    )["train"]

    augemented_dataset = load_dataset(
        "json",
        data_files=augment_json_path,
        cache_dir=None,
        token=None,
    )["train"]

    # updated_dataset = small_dataset.map(lambda example, idx: {'sentence2': f'{idx}: ' + example['sentence2']}, with_indices=True)
    def map_row(row, i):
        row["translation"]["zh"] = zh_translations[i]
        return row
    
    augemented_dataset = augemented_dataset.map(map_row, with_indices=True)

    new_dataset = concatenate_datasets([augemented_dataset, train_dataset], axis=0)
    
    with open(concatenated_train_json_path, mode="w") as f:
        for row in new_dataset:
            json.dump(row, f)
            f.write("\n")
 
    return new_dataset

if __name__ == "__main__":
    DATA_DIR = "../data"
   
    ### STEP 1: Generate augmented English sentences
    # input_path =  os.path.join(DATA_DIR, "train.json")
    # output_path = os.path.join(DATA_DIR, "augmented.json")
    # ooc_words_path = os.path.join(DATA_DIR, "ooc_words.json")

    # augmented_sentences = main(
    #     input_json_path=input_path, 
    #     ooc_words_txt_path=ooc_words_path, 
    #     output_json_path=output_path,
    #     augment_func=lambda inputs, ooc_words: constituency_sub_augmentation(inputs, ooc_words, n_generate=20000)
    # ) 

    ### STEP 2
    # Run inference to get Chinese translations in generate_predictions.txt
    """
    python ft.py \
        --model_name_or_path ./opus_baseline \
        --do_predict \
        --source_lang en \
        --target_lang zh \
        --max_source_length 512 \
        --test_file ./data/augmented.json \
        --output_dir ./opus_baseline \
        --per_device_train_batch_size=4 \
        --per_device_eval_batch_size=4 \
        --predict_with_generate
    """

    ### STEP 3: Generate concatenated dataset
    # train_json_path = os.path.join(DATA_DIR, "train.json")
    # augment_json_path = os.path.join(DATA_DIR, "augmented-3.json")
    # translations_txt_path = os.path.join(OUTPUT_DIR, "generated_predictions.txt")
    # concatenated_train_json_path = os.path.join(DATA_DIR, "augmented_train.json")

    # augmented_dataset = augment_training_data(
    #     train_json_path=train_json_path,
    #     augment_json_path=augment_json_path,
    #     translations_txt_path=translations_txt_path,
    #     concatenated_train_json_path=concatenated_train_json_path
    # )
    # print(augmented_dataset)

    ### STEP 4
    # Run model with augmented data
    """
    python ft.py \
        --model_name_or_path Helsinki-NLP/opus-mt-en-zh \
        --do_train \
        --do_eval \
        --do_predict \
        --source_lang en \
        --target_lang zh \
        --max_source_length 512 \
        --num_train_epochs 3 \
        --save_total_limit 2 \
        --eval_steps 5000 \
        --logging_steps 5000 \
        --save_steps 5000 \
        --evaluation_strategy steps \
        --train_file ./data/augmented_train.json \
        --test_file ./data/test.json \
        --validation_file ./data/validation.json \
        --output_dir ./const-sub-20k-min-3-words \
        --per_device_train_batch_size=4 \
        --per_device_eval_batch_size=4 \
        --predict_with_generate
    """