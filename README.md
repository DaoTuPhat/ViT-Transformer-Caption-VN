# Vietnamese Image Captioning with Vision Transformer 🇻🇳

**ViCap** is a Deep Learning project designed to build a model capable of automatically generating natural Vietnamese descriptions (captions) for input images.

The project leverages the robust feature extraction capabilities of **Vision Transformer (ViT)** combined with the language processing power of a **Transformer Decoder**. Specifically, this model is engineered to address the **Low-Resource Image Captioning** challenge, utilizing a training dataset of only approximately 21,000 images.

## 🏗️ Model Architecture
The model follows a standard Encoder-Decoder architecture:

1.  **Image Encoder (ViT-Base):** Splits the image into 16x16 patches, flattens them, and processes them through Self-Attention layers.
    * *Input:* `(Batch, 3, 224, 224)`
    * *Output:* `(Batch, 197, 768)` (including the CLS token).
2.  **Bridge Layer:** A Linear layer that projects feature dimensions from 768 (ViT) down to 512 (Decoder).
3.  **Text Decoder (Transformer):** Auto-regressive generation with Causal Masking.
    * *Input:* Tokenized Vietnamese captions.
    * *Output:* Probability distribution over the vocabulary (~4000 words).

## 📂 Dataset
Dataset Link: https://www.kaggle.com/datasets/easterharry/info-retrieval-v2

The model is trained on a consolidated dataset of approximately **21,000 images**, sourced from:
* **Flickr8k** (Vietnamese translated version).
* **UIT-ViIC** (UIT's Vietnamese Image Captioning dataset).
* **KTVIC**.

* **Vocabulary Size:** ~4,000 common Vietnamese words.
* **Tokenizer:** Word-level tokenization.

## 🛠️ Installation & Environment
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/DaoTuPhat/ViT-Transformer-Caption-VN.git](https://github.com/DaoTuPhat/ViT-Transformer-Caption-VN.git)
    cd ViT-Transformer-Caption-VN
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Prepare Data:**
    Place your images in the `data/` directory and ensure the file paths in the configuration CSV are correct.