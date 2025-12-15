import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class EvalDataset(Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, cap_ids, length = self.base[idx]
        row = self.base.df.iloc[idx]
        image_key = row["image_filename"]
        ref_caption_tok = row["caption_tok"]
        return image, cap_ids, length, image_key, ref_caption_tok


def collate_fn_eval(batch):
    images = [b[0] for b in batch]
    keys = [b[3] for b in batch]
    refs = [b[4] for b in batch]

    images = torch.stack(images, dim=0)
    return images, keys, refs


def decode_tokens(token_ids, itos, eos_idx, bos_idx, pad_idx):
    tokens = []
    for token_id in token_ids:
        token_id = int(token_id)
        if token_id == eos_idx:
            break
        if token_id == bos_idx or token_id == pad_idx:
            continue

        word = itos.get(token_id, "<unk>")
        tokens.append(word)

    return tokens


def generate_captions_for_dataset(
    model,
    base_dataset,
    device,
    bos_idx,
    eos_idx,
    pad_idx,
    max_len=40,
    batch_size=64,
    beam_size=3,
):
    eval_ds = EvalDataset(base_dataset)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_eval,
        num_workers=2,
    )

    model.eval()
    all_references = {}
    all_hypotheses = {}
    itos = base_dataset.itos

    with torch.no_grad():
        pbar = tqdm(eval_loader, desc="Evaluating")

        for images, keys, refs in pbar:
            images = images.to(device)

            for i, key in enumerate(keys):
                # Collect references
                ref_str = refs[i]
                ref_tokens = ref_str.split() if isinstance(ref_str, str) else ref_str

                if key not in all_references:
                    all_references[key] = []
                all_references[key].append(ref_tokens)

                # Generate hypothesis
                if key not in all_hypotheses:
                    img_tensor = images[i].unsqueeze(0)  # (1, 3, 224, 224)
                    generated_seq = model.generate_beam(
                        img_tensor,
                        bos_idx=bos_idx,
                        eos_idx=eos_idx,
                        max_len=max_len,
                        beam_size=beam_size,
                    )  # (1, max_len)
                    hyp_tokens = decode_tokens(
                        generated_seq, itos, eos_idx, bos_idx, pad_idx
                    )
                    all_hypotheses[key] = hyp_tokens

    return all_references, all_hypotheses



import nltk
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score

nltk.download("wordnet")
nltk.download("omw-1.4")


def compute_bleu_meteor(all_references, all_hypotheses):
    """
    all_references: {img: [[ref1_tokens], [ref2_tokens], ...]}
    all_hypotheses: {img: [hyp_tokens]}
    """
    # BLEU
    keys = [k for k in all_hypotheses.keys() if k in all_references]
    list_of_refs = [all_references[k] for k in keys]
    hyps = [all_hypotheses[k] for k in keys]

    smoothie = SmoothingFunction().method1

    bleu1 = corpus_bleu(
        list_of_refs, hyps, weights=(1.0, 0, 0, 0), smoothing_function=smoothie
    )
    bleu2 = corpus_bleu(
        list_of_refs, hyps, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie
    )
    bleu3 = corpus_bleu(
        list_of_refs,
        hyps,
        weights=(1 / 3, 1 / 3, 1 / 3, 0),
        smoothing_function=smoothie,
    )
    bleu4 = corpus_bleu(
        list_of_refs,
        hyps,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothie,
    )

    # METEOR
    meteor_scores = []
    for refs_tok, hyp_tok in zip(list_of_refs, hyps):
        meteor_scores.append(meteor_score(refs_tok, hyp_tok))
    meteor_avg = sum(meteor_scores) / len(meteor_scores)

    return {
        "BLEU-1": bleu1,
        "BLEU-2": bleu2,
        "BLEU-3": bleu3,
        "BLEU-4": bleu4,
        "METEOR": meteor_avg,
    }



from pycocoevalcap.cider.cider import Cider


def compute_cider(all_references, all_hypotheses):
    """
    all_hypotheses: dict
        { key: [tok1, tok2, ...] }  # 1 caption dự đoán (tokenized)
    all_references: dict
        { key: [[tok1, tok2, ...], [tok1, tok2, ...], ...] }  # nhiều caption ref (tokenized)
    """
    gts = {}
    res = {}
    keys = [k for k in all_hypotheses.keys() if k in all_references]

    for i, key in enumerate(keys):
        # hypothesis: list token -> str
        hyp_tok = all_hypotheses[key]
        hyp_str = " ".join(hyp_tok)
        res[i] = [hyp_str]

        # references: list[list token] -> list[str]
        refs_tok = all_references[key]  # [[...], [...], ...]
        refs_str = [" ".join(r) for r in refs_tok]
        gts[i] = refs_str

    cider_scorer = Cider()
    score, scores = cider_scorer.compute_score(gts, res)
    return score
