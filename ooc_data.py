import json

pairs = []
with open("./data/en.txt") as en:
    with open("./data/zh.txt") as zh:
        with open("./data/ooc_test.json", mode="w+") as f:
            for (en, zh) in zip(en.readlines(), zh.readlines()):
                pair = {}
                pair["translation"] = {}
                pair["translation"]["en"] = en.strip()
                pair["translation"]["zh"] = zh.strip()
                json.dump(pair, f)
                f.write("\n")
    