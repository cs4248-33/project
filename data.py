import os
from datasets import load_dataset

data_dir = "./data"
splits = ["train", "test", "validation"]
dataset = load_dataset(
    "iwslt2017", "iwslt2017-en-zh",
    data_dir=data_dir, 
    cache_dir=data_dir 
)

for split in splits:
    path = os.path.join(data_dir, f"{split}.json")

    if os.path.exists(path):
        print("{} already saved in {}".format(split, path))
        continue

    dataset[split].to_json(path)

