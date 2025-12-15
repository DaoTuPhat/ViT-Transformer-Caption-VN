import pandas as pd
import numpy as np
from tqdm import tqdm
from underthesea import word_tokenize
from collections import Counter
import json


# SPLIT DATASET
def split_dataset(CLEANED_CSV_PATH, SPIT_CSV_PATH):
    df = pd.read_csv(CLEANED_CSV_PATH, sep=";")
    n_captions = len(df)
    print("Total captions:", n_captions)

    unique_images = df["image_filename"].unique()
    n_images = len(unique_images)
    print("Number of images:", n_images)

    # Shuffle images with fixed seed
    rng = np.random.default_rng(42)
    rng.shuffle(unique_images)

    # Calculate number for each split (80/10/10)
    n_train = int(0.8 * n_images)
    n_val = int(0.1 * n_images)
    n_test = n_images - n_train - n_val

    train_imgs = set(unique_images[:n_train])
    val_imgs = set(unique_images[n_train : n_train + n_val])
    test_imgs = set(unique_images[n_train + n_val :])

    print("Train images:", len(train_imgs))
    print("Validation images:", len(val_imgs))
    print("Test images:", len(test_imgs))

    def assign_split(img_name):
        if img_name in train_imgs:
            return "train"
        elif img_name in val_imgs:
            return "val"
        else:
            return "test"

    # Assign new split according to image_filename
    df["split"] = df["image_filename"].map(assign_split)
    print(df["split"].value_counts())

    df.to_csv(SPIT_CSV_PATH, sep=";", index=False)
    print(f"Saved: {SPIT_CSV_PATH}")


# PROCESS CAPTION
def process_captions(SPIT_CSV_PATH, PROCESSED_CSV_PATH):
    df = pd.read_csv(SPIT_CSV_PATH, sep=";")

    def tokenize_vi(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.strip()
        if not text:
            return ""
        return word_tokenize(text, format="text")

    # Process captions
    tqdm.pandas(desc="Tokenizing captions")
    df["caption_tok"] = df["caption"].progress_apply(tokenize_vi)

    df.to_csv(PROCESSED_CSV_PATH, sep=";", index=False)
    print(f"Saved: {PROCESSED_CSV_PATH}")


# CREATE VIETNAMESE VOCABULARY
def create_vocabulary(PROCESSED_CSV_PATH, VOCAB_PATH, MIN_FREQ=5):
    # 1. Get only train
    df = pd.read_csv(PROCESSED_CSV_PATH, sep=";")
    train_df = df[df["split"] == "train"]

    # 2. Count token frequency
    counter = Counter()

    for line in train_df["caption_tok"]:
        if not isinstance(line, str):
            continue
        tokens = line.strip().split()
        counter.update(tokens)

    # 3. Initialize vocab with special tokens
    stoi = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
    }

    # 4. Add regular tokens (filter by frequency)
    idx = len(stoi)
    for token, freq in counter.most_common():
        if freq < MIN_FREQ:
            continue
        if token in stoi:
            continue
        stoi[token] = idx
        idx += 1

    print("Number of tokens in vocab:", len(stoi))

    # 5. Save vocab to JSON file
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(stoi, f, ensure_ascii=False, indent=2)
    print(f"Saved: {VOCAB_PATH}")


if __name__ == "__main__":
    CSV_PATH = "data/final_combined_ds.csv"
    SPIT_CSV_PATH = "data/final_combined_ds_with_split.csv"
    PROCESSED_CSV_PATH = "data/final_combined_ds_tokenized.csv"
    VOCAB_PATH = "data/vocab_vi_underthesea.json"

    # SPLITTING
    split_dataset(CSV_PATH, SPIT_CSV_PATH)

    # PROCESSING CAPTIONS
    process_captions(SPIT_CSV_PATH, PROCESSED_CSV_PATH)

    # CREATE VOCABULARY
    create_vocabulary(PROCESSED_CSV_PATH, VOCAB_PATH, MIN_FREQ=5)
