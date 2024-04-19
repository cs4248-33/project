import json
import argparse
from datasets import load_dataset, concatenate_datasets

def main(
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_json_path", 
        help="Path to JSONLINES file containing the original training data",
        type=str
    )
    parser.add_argument(
        "--augment_json_path", 
        help="Path to JSONLINES file containing the augmented data",
        type=str
    )
    parser.add_argument(
        "--translations_txt_path", 
        help="Path to .txt file ontaining the translations of each augmented sentence",
        type=str
    )
    parser.add_argument(
        "--concatenated_train_json_path", 
        help="Path to JSONLINES file which will be overwritten with the final training set, combining both the augmented and original sentence pairs.",
        type=str
    )

    args = parser.parse_args()

    assert args.train_json_path is not None
    assert args.augment_json_path is not None
    assert args.translations_txt_path is not None
    assert args.concatenated_train_json_path is not None 

    train_json_path = args.train_json_path
    augment_json_path = args.augment_json_path
    translations_txt_path = args.translations_txt_path
    concatenated_train_json_path = args.concatenated_train_json_path

    print("Combining translations with augmented data...")
    augmented_dataset = main(
        train_json_path=train_json_path,
        augment_json_path=augment_json_path,
        translations_txt_path=translations_txt_path,
        concatenated_train_json_path=concatenated_train_json_path
    )