import os
import json
import pandas as pd
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def load_vocab(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        # {token: idx}
        stoi = json.load(f) 
        
    # {idx: token}
    itos = {int(idx): tok for tok, idx in stoi.items()}
    return stoi, itos


class ViImageCaptionDataset(Dataset):
    def __init__(
        self,
        csv_path,  
        vocab_path,  
        data_root,  
        split="train",  
        transform=None,  
        max_len=40,  
        sep=";",  
        source_to_dir=None,  
    ):
        super().__init__()
        self.csv_path = csv_path
        self.vocab_path = vocab_path
        self.data_root = data_root
        self.split = split
        self.max_len = max_len
        self.sep = sep

        # Load dataframe
        df = pd.read_csv(csv_path, sep=sep)
        df = df[df["split"] == split].reset_index(drop=True)
        self.df = df

        # Load vocab
        self.stoi, self.itos = load_vocab(vocab_path)
        self.pad_idx = self.stoi["<pad>"]
        self.bos_idx = self.stoi["<bos>"]
        self.eos_idx = self.stoi["<eos>"]
        self.unk_idx = self.stoi["<unk>"]

        # Map dataset_source to directory
        if source_to_dir is None:
            #  - "Flickr"       -> data/flickr
            #  - "KTVIC"        -> data/ktvic
            #  - "UIT-OpenViIC" -> data/UIT-OpenViIC
            source_to_dir = {
                "Flickr": os.path.join(data_root, "flickr"),
                "KTVIC": os.path.join(data_root, "ktvic"),
                "UIT-OpenViIC": os.path.join(data_root, "UIT-OpenViIC"),
            }
        self.source_to_dir = source_to_dir

        # Transform
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
        else:
            self.transform = transform

    def __len__(self):
        return len(self.df)

    def numericalize_caption(self, caption_tok: str):
        if not isinstance(caption_tok, str):
            tokens = []
        else:
            tokens = caption_tok.strip().split()

        ids = [self.bos_idx]
        for tok in tokens:
            ids.append(self.stoi.get(tok, self.unk_idx))
        ids.append(self.eos_idx)

        if len(ids) > self.max_len:
            ids = ids[: self.max_len]
            if ids[-1] != self.eos_idx:
                ids[-1] = self.eos_idx

        length = len(ids)
        ids = torch.tensor(ids, dtype=torch.long)
        return ids, length

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        src = row["dataset_source"]
        if src not in self.source_to_dir:
            raise ValueError(f"Not found in source_to_dir: {src}")

        img_dir = self.source_to_dir[src]
        img_path = os.path.join(img_dir, row["image_filename"])
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Error opening image {img_path}: {e}. Using black image instead.")
            from PIL import Image as _Image
            image = _Image.new("RGB", (224, 224), (0, 0, 0))

        image = self.transform(image)
        caption_tok = row["caption_tok"]
        caption_ids, length = self.numericalize_caption(caption_tok)

        return image, caption_ids, length


def build_collate_fn(pad_idx):
    def collate_fn(batch):
        images, captions, lengths = zip(*batch)
        images = torch.stack(images, dim=0)
        lengths = torch.tensor(lengths, dtype=torch.long)
        max_len = lengths.max().item()

        padded_captions = torch.full(
            (len(captions), max_len),
            fill_value=pad_idx,
            dtype=torch.long,
        )

        for i, cap in enumerate(captions):
            end = cap.size(0)
            padded_captions[i, :end] = cap[:max_len]

        return images, padded_captions, lengths

    return collate_fn


def create_dataloaders(
    csv_path,
    vocab_path,
    data_root,
    batch_size=64,
    max_len=40,
    num_workers=0,
    sep=";",
):
    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_dataset = ViImageCaptionDataset(
        csv_path=csv_path,
        vocab_path=vocab_path,
        data_root=data_root,
        split="train",
        transform=train_transform,
        max_len=max_len,
        sep=sep,
    )

    val_dataset = ViImageCaptionDataset(
        csv_path=csv_path,
        vocab_path=vocab_path,
        data_root=data_root,
        split="val",
        transform=val_transform,
        max_len=max_len,
        sep=sep,
    )

    test_dataset = ViImageCaptionDataset(
        csv_path=csv_path,
        vocab_path=vocab_path,
        data_root=data_root,
        split="test",
        transform=val_transform,
        max_len=max_len,
        sep=sep,
    )

    pad_idx = train_dataset.pad_idx
    collate_fn = build_collate_fn(pad_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        val_dataset,
        test_dataset,
    )
