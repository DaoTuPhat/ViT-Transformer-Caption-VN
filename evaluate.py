import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import nltk
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.cider.cider import Cider

# Tải dữ liệu WordNet
nltk.download('wordnet')

# Tải gói hỗ trợ đa ngôn ngữ (thường cần cho bản NLTK mới)
nltk.download('omw-1.4')


class EvalDataset(Dataset):
    """
    Wrapper quanh dataset gốc để lấy thêm:
      - image_key (dùng image_filename làm ID)
      - ref_caption_tok (caption đã tokenize)
    """
    def __init__(self, base_dataset):
        self.base = base_dataset        # train_ds / val_ds / test_ds


    def __len__(self):
        return len(self.base)


    def __getitem__(self, idx):
        image, cap_ids, length = self.base[idx]
        row = self.base.df.iloc[idx]
        image_key = row["image_filename"]     
        ref_caption_tok = row["caption_tok"]
        return image, cap_ids, length, image_key, ref_caption_tok
    

def collate_fn_eval(batch):
    images  = [b[0] for b in batch]
    keys    = [b[3] for b in batch]  # image_key
    refs    = [b[4] for b in batch]  # ref_caption_tok

    images = torch.stack(images, dim=0)
    return images, keys, refs


def generate_captions_for_dataset(model, base_dataset, device, bos_idx, eos_idx,
                                  max_len=30, batch_size=64):
    eval_ds = EvalDataset(base_dataset)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_eval,
    )

    model.eval()
    all_references = {}
    all_hypotheses = {}

    with torch.no_grad():
        pbar = tqdm(eval_loader, desc="Generating", leave=False)
        for images, keys, refs in pbar:
            images = images.to(device)
            gen_ids = model.generate(
                images,
                bos_idx=bos_idx,
                eos_idx=eos_idx,
                max_len=max_len,
            )  # (B, max_len)
            gen_ids = gen_ids.cpu().numpy()

            for i, key in enumerate(keys):
                hyp_tokens = []
                for idx in gen_ids[i]:
                    idx = int(idx)
                    if idx == eos_idx:
                        break
                    tok = base_dataset.itos.get(idx, "<unk>")
                    if tok not in ["<bos>", "<pad>"]:
                        hyp_tokens.append(tok)

                if key not in all_hypotheses:
                    all_hypotheses[key] = hyp_tokens

                ref_str = refs[i]
                ref_tokens = ref_str.split() if isinstance(ref_str, str) else []
                all_references.setdefault(key, []).append(ref_tokens)

    return all_references, all_hypotheses


def compute_bleu_meteor(all_references, all_hypotheses):
    """
    all_references: {img: [[ref1_tokens], [ref2_tokens], ...]}
    all_hypotheses: {img: [hyp_tokens]}
    """
    keys = [k for k in all_hypotheses.keys() if k in all_references]

    list_of_refs = [all_references[k] for k in keys]  # mỗi phần tử: list các ref (token list)
    hyps         = [all_hypotheses[k] for k in keys]  # mỗi phần tử: 1 hyp (token list)

    smoothie = SmoothingFunction().method1

    bleu1 = corpus_bleu(list_of_refs, hyps,
                        weights=(1.0, 0, 0, 0),
                        smoothing_function=smoothie)
    bleu2 = corpus_bleu(list_of_refs, hyps,
                        weights=(0.5, 0.5, 0, 0),
                        smoothing_function=smoothie)
    bleu3 = corpus_bleu(list_of_refs, hyps,
                        weights=(1/3, 1/3, 1/3, 0),
                        smoothing_function=smoothie)
    bleu4 = corpus_bleu(list_of_refs, hyps,
                        weights=(0.25, 0.25, 0.25, 0.25),
                        smoothing_function=smoothie)

    # METEOR: trung bình trên từng ảnh
    meteor_scores = []
    for refs_tok, hyp_tok in zip(list_of_refs, hyps):
        # refs_str = [" ".join(r) for r in refs_tok]
        # hyp_str  = " ".join(hyp_tok)
        meteor_scores.append(meteor_score(refs_tok, hyp_tok))
    meteor_avg = sum(meteor_scores) / len(meteor_scores)

    return {
        "BLEU-1": bleu1,
        "BLEU-2": bleu2,
        "BLEU-3": bleu3,
        "BLEU-4": bleu4,
        "METEOR": meteor_avg,
    }


def compute_cider(all_references, all_hypotheses):
    """
    all_hypotheses: dict
        { key: [tok1, tok2, ...] }  # 1 caption dự đoán (tokenized)
    all_references: dict
        { key: [[tok1, tok2, ...], [tok1, tok2, ...], ...] }  # nhiều caption ref (tokenized)
    """
    gts = {}
    res = {}

    # chỉ lấy những key có cả hyp lẫn ref
    keys = [k for k in all_hypotheses.keys() if k in all_references]

    for i, key in enumerate(keys):
        # hypothesis: list token -> string
        hyp_tok = all_hypotheses[key]          # ["một", "con", "chó", ...]
        hyp_str = " ".join(hyp_tok)            # "một con chó ..."

        # res[i] phải là list[str]
        res[i] = [hyp_str]

        # references: list[list token] -> list[str]
        refs_tok = all_references[key]         # [[...], [...], ...]
        refs_str = [" ".join(r) for r in refs_tok]

        # gts[i] cũng là list[str]
        gts[i] = refs_str

    cider_scorer = Cider()
    score, scores = cider_scorer.compute_score(gts, res)
    return score  # CIDEr trung bình
