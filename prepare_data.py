import pandas as pd
import numpy as np
from tqdm import tqdm
from underthesea import word_tokenize
from collections import Counter
import json


# 1. DATA CLEANING
def clean_captions(CSV_PATH, CLEANED_CSV_PATH):
    df = pd.read_csv(CSV_PATH, sep=";")

    def replace_semicolon(text):
        if not isinstance(text, str):
            return text
        return text.replace(";", ",")

    df["caption"] = df["caption"].apply(replace_semicolon)
    df["caption_en"] = df["caption_en"].apply(replace_semicolon)

    df.to_csv(CLEANED_CSV_PATH, sep=";", index=False)
    print(f"Đã lưu: {CLEANED_CSV_PATH}")


# 2. CHIA DATASET
def split_dataset(CLEANED_CSV_PATH, SPIT_CSV_PATH):
    # Đọc file gốc (dùng ; làm separator)
    df = pd.read_csv(CLEANED_CSV_PATH, sep=";")
    n_captions = len(df)
    print("Tổng số caption:", n_captions)

    unique_images = df["image_filename"].unique()
    n_images = len(unique_images)
    print("Số ảnh:", n_images)


    # Shuffle ảnh với seed cố định
    rng = np.random.default_rng(42)
    rng.shuffle(unique_images)


    # Tính số lượng cho mỗi split (80/10/10)
    n_train = int(0.8 * n_images)
    n_val   = int(0.1 * n_images)
    n_test  = n_images - n_train - n_val

    train_imgs = set(unique_images[:n_train])
    val_imgs   = set(unique_images[n_train:n_train + n_val])
    test_imgs  = set(unique_images[n_train + n_val:])

    print("Ảnh train:", len(train_imgs))
    print("Ảnh val:", len(val_imgs))
    print("Ảnh test:", len(test_imgs))

    def assign_split(img_name):
        if img_name in train_imgs:
            return "train"
        elif img_name in val_imgs:
            return "val"
        else:
            return "test"
    
    # Gán split mới theo image_filename
    df["split"] = df["image_filename"].map(assign_split)
    print(df["split"].value_counts())

    # Lưu ra file mới
    df.to_csv(SPIT_CSV_PATH, sep=";", index=False)
    print(f"Đã lưu: {SPIT_CSV_PATH}")


# 3. XỬ LÝ CAPTION 
def process_captions(SPIT_CSV_PATH, PROCESSED_CSV_PATH):
    # Đọc file gốc (dùng ; làm separator)
    df = pd.read_csv(SPIT_CSV_PATH, sep=";")

    def tokenize_vi(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.strip()
        if not text:
            return ""
        # format="text" => trả về string, các token cách nhau bằng space
        return word_tokenize(text, format="text")

    # Xử lý caption
    tqdm.pandas(desc="Tokenizing captions")
    df["caption_tok"] = df["caption"].progress_apply(tokenize_vi)

    # Lưu ra file mới
    df.to_csv(PROCESSED_CSV_PATH, sep=";", index=False)
    print(f"Đã lưu: {PROCESSED_CSV_PATH}")


# 4. Tạo bộ vocabulary tiếng việt
def create_vocabulary(PROCESSED_CSV_PATH, VOCAB_PATH, MIN_FREQ=5):
    # 1. Lấy chỉ train
    df = pd.read_csv(PROCESSED_CSV_PATH, sep=";")
    train_df = df[df["split"] == "train"]

    # 2. Đếm tần suất token
    counter = Counter()

    for line in train_df["caption_tok"]:
        if not isinstance(line, str):
            continue
        tokens = line.strip().split()
        counter.update(tokens)

    # 3. Khởi tạo vocab với token đặc biệt
    stoi = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
    }

    # 4. Thêm token thường (lọc theo tần suất)
    idx = len(stoi)
    for token, freq in counter.most_common():
        if freq < MIN_FREQ:
            continue
        if token in stoi:
            continue
        stoi[token] = idx
        idx += 1

    print("Số lượng token trong vocab:", len(stoi))

    # 5. Lưu vocab ra file JSON
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(stoi, f, ensure_ascii=False, indent=2)

    print(f"Đã lư: {VOCAB_PATH}")


if __name__ == "__main__":
    CSV_PATH = "phat/final_combined_ds.csv"
    CLEANED_CSV_PATH = "phat/final_combined_ds_cleaned.csv"
    SPIT_CSV_PATH = "phat/final_combined_ds_with_split.csv"
    PROCESSED_CSV_PATH = "phat/final_combined_ds_tokenized.csv"
    VOCAB_PATH = "phat/vocab_vi_underthesea.json"

    # CLEANING
    clean_captions(CSV_PATH, CLEANED_CSV_PATH)

    # SPLITTING
    split_dataset(CLEANED_CSV_PATH, SPIT_CSV_PATH)

    # PROCESSING CAPTIONS
    process_captions(SPIT_CSV_PATH, PROCESSED_CSV_PATH)

    # CREATE VOCABULARY
    create_vocabulary(PROCESSED_CSV_PATH, VOCAB_PATH, MIN_FREQ=5)
    
    