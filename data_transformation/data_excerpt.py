from datetime import datetime
import json
import random
import argparse

# This script just creates some random data for testing purposes.
# Pass --input=raw to get the raw data and --input=processed to get the already processed data to sample from

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="raw", nargs="?",
                    help="What data to import (raw/processed) (default:raw)")

parser.add_argument("--output", default="sample", nargs="?",
                    help="What file name to output to /data directory (default: sample_1k_<TYPE>_<TIMESTAMP>.json)")

parser.add_argument("--size", default=1000, nargs="?", type=int,
                    help="Size of the sample (default: 1000)")

args, unknown = parser.parse_known_args()
if args.input == "raw":
    ending = "data_bis_19.json"
else:
    ending = "data_processed.json"

import_input = "../data/" + ending
output_file = "../data/" + args.output + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".json"
size = args.size

with open(import_input, "r") as file:
    data = json.load(file)

sampled_elements = random.sample(data, size)

with open(output_file, "w") as outputfile:
    json.dump(sampled_elements, outputfile, ensure_ascii=False)