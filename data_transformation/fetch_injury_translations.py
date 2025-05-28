import json

from deepl import DeepLClient
from dotenv import load_dotenv
import os
from tqdm import tqdm

load_dotenv()

with open("../data/mapping/injury_to_category.json", "r") as f:
    data = json.load(f)

injuries = list(data.keys())

auth_key = os.getenv("deepl_auth_key")
deeplClient = DeepLClient(auth_key)

german_to_english_injury = {}
for injury in tqdm(injuries):
    translation = deeplClient.translate_text(text=injury, source_lang="DE", target_lang="EN-US")
    german_to_english_injury[injury] = str(translation)


with open("../data/mapping/injury_translations.json", "w") as f:
    json.dump(german_to_english_injury, f, ensure_ascii=False)
