from datasets import load_dataset
import pandas as pd
import os

dataset = load_dataset("iwslt2017", "iwslt2017-en-zh")
splits = ["train", "test", "validation"]

for split in splits:
    path = '../data/{}.csv'.format(split)
    
    if os.path.exists(path):
        print("{} already saved in {}".format(split, path))
        continue
    
    data = dataset[split]
    df = pd.DataFrame({
        "en": [item['translation']['en'] for item in data],
        "zh": [item['translation']['zh'] for item in data]
    })

    df.to_csv(path, index=False)

    print("Saved {} dataset to {}".format(split, path))