import json

with open("../data/data_processed_jan.json", "r") as f:
    data = json.load(f)

with open("../data/mapping/injury_translations.json", "r") as f:
    translations = json.load(f)

for _, item in enumerate(data):
    if item["injury"] is not None:
        item["injury"] = translations[item["injury"]]


with open("../data/data_processed.json", "w") as f:
    json.dump(data, f, ensure_ascii=False)