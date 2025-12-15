import torch
import torch.nn as nn
from transformers import ViTModel


class Attention(nn.Module):
    """
    Attention Network (Soft Attention)
    """
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        """
        encoder_out: (Batch, 197, Encoder_Dim)
        decoder_hidden: (Batch, Decoder_Dim)
        """
        att1 = self.encoder_att(encoder_out)  # (B, 197, Att_Dim)
        att2 = self.decoder_att(decoder_hidden)  # (B, Att_Dim)

        # (B, 197, Att_Dim) + (B, 1, Att_Dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1)))  # (B, 197, 1)

        alpha = self.softmax(att)  # (B, 197, 1)
        attention_weighted_encoding = (encoder_out * alpha).sum(
            dim=1
        )  # (B, Encoder_Dim)

        return attention_weighted_encoding, alpha


class ViT_LSTM_Attention(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=512,
        hidden_dim=512,
        attention_dim=256,
        unfreeze_layers=2,
        dropout=0.3,
    ):
        super(ViT_LSTM_Attention, self).__init__()

        # ENCODER (ViT)
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.encoder_dim = 768

        # Freeze & Unfreeze logic
        for param in self.vit.parameters():
            param.requires_grad = False
        if unfreeze_layers > 0:
            for layer in self.vit.encoder.layer[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            for param in self.vit.layernorm.parameters():
                param.requires_grad = True

        # ATTENTION
        self.attention = Attention(self.encoder_dim, hidden_dim, attention_dim)

        # DECODER (LSTM)
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.lstm_cell = nn.LSTMCell(embed_dim + self.encoder_dim, hidden_dim)

        self.init_h = nn.Linear(self.encoder_dim, hidden_dim)
        self.init_c = nn.Linear(self.encoder_dim, hidden_dim)

        self.f_beta = nn.Linear(hidden_dim, self.encoder_dim)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, images, captions):
        """
        images: (B, 3, 224, 224)
        captions: (B, Seq_Len) - Chứa cả <bos> và <eos>
        """
        device = images.device

        # Encoder ảnh
        with torch.no_grad():
            encoder_out = self.vit(
                pixel_values=images
            ).last_hidden_state  # (B, 197, 768)

        batch_size = encoder_out.size(0)
        vocab_size = self.vocab_size

        # Khởi tạo LSTM state từ trung bình ảnh
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)

        # Xử lý Captions (Bỏ <eos> ở cuối vì ta dự đoán nó)
        embeddings = self.embedding(captions)  # (B, Seq_Len, Embed_Dim)

        # Tensor chứa kết quả: (B, Seq_Len-1, Vocab)
        seq_len = captions.size(1) - 1
        predictions = torch.zeros(batch_size, seq_len, vocab_size).to(device)

        # Decoder Loop
        for t in range(seq_len):
            # Lấy attention context từ hidden state cũ
            attention_weighted_encoding, alpha = self.attention(encoder_out, h)

            # Quyết định mức độ quan trọng của ảnh
            gate = torch.sigmoid(self.f_beta(h))
            attention_weighted_encoding = gate * attention_weighted_encoding

            # LSTM Step
            # Input: Embedding từ bước t + Context Vector từ ảnh
            lstm_input = torch.cat(
                [embeddings[:, t, :], attention_weighted_encoding], dim=1
            )

            # Update LSTM
            h, c = self.lstm_cell(lstm_input, (h, c))

            # Predict next word
            preds = self.fc_out(self.dropout(h))
            predictions[:, t, :] = preds

        return predictions

    def generate_beam(self, images, bos_idx, eos_idx, max_len=40, beam_size=3):
        self.eval()
        device = images.device
        batch_size = images.size(0)

        with torch.no_grad():
            # Encoder
            encoder_out = self.vit(
                pixel_values=images
            ).last_hidden_state  # (1, 197, 768)
            mean_encoder_out = encoder_out.mean(dim=1)
            encoder_out = encoder_out.expand(beam_size, -1, -1)  # (Beam, 197, 768)

            k_prev_words = torch.tensor([[bos_idx]] * beam_size, dtype=torch.long).to(
                device
            )  # (Beam, 1)
            seqs = k_prev_words  # (Beam, 1)
            top_k_scores = torch.zeros(beam_size, 1).to(device)

            # Init Hidden States
            h = self.init_h(mean_encoder_out).expand(beam_size, -1)  # (Beam, Hidden)
            c = self.init_c(mean_encoder_out).expand(beam_size, -1)

            completed_seqs = []
            completed_scores = []

            for step in range(max_len):
                last_word_idx = seqs[:, -1]
                embeddings = self.embedding(last_word_idx)  # (Beam, Embed_Dim)

                # Attention
                attention_weighted_encoding, alpha = self.attention(encoder_out, h)
                gate = torch.sigmoid(self.f_beta(h))
                attention_weighted_encoding = gate * attention_weighted_encoding

                # LSTM
                lstm_input = torch.cat([embeddings, attention_weighted_encoding], dim=1)
                h, c = self.lstm_cell(lstm_input, (h, c))

                # Predict
                scores = self.fc_out(h)  # (Beam, Vocab)
                log_probs = torch.log_softmax(scores, dim=1)

                # Cộng điểm tích lũy
                scores = top_k_scores.expand_as(log_probs) + log_probs  # (Beam, Vocab)

                # Flatten để lấy top k trên toàn bộ (Beam * Vocab)
                if step == 0:
                    # Bước đầu tiên tất cả beam giống nhau, chỉ lấy top k của beam 0
                    top_k_scores, top_k_words = scores[0].topk(beam_size, 0, True, True)
                else:
                    top_k_scores, top_k_words = scores.view(-1).topk(
                        beam_size, 0, True, True
                    )

                # Convert indices về (Beam Index, Word Index)
                prev_beam_indices = top_k_words // self.vocab_size
                next_word_indices = top_k_words % self.vocab_size

                # Update sequences
                seqs = torch.cat(
                    [seqs[prev_beam_indices], next_word_indices.unsqueeze(1)], dim=1
                )

                # Kiểm tra EOS
                incomplete_inds = []
                for i, word_idx in enumerate(next_word_indices):
                    if word_idx == eos_idx:
                        completed_seqs.append(seqs[i])
                        completed_scores.append(top_k_scores[i])
                    else:
                        incomplete_inds.append(i)

                # Nếu tất cả đã xong hoặc beam size giảm về 0
                if len(incomplete_inds) == 0:
                    break

                # Giữ lại các beam chưa xong để đi tiếp
                seqs = seqs[incomplete_inds]
                h = h[prev_beam_indices[incomplete_inds]]
                c = c[prev_beam_indices[incomplete_inds]]
                encoder_out = encoder_out[
                    prev_beam_indices[incomplete_inds]
                ]  # Cần reorder cả encoder_out
                top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)

                # Nếu số lượng beam active giảm , ta giảm beam_size thực tế xuống
                beam_size = len(incomplete_inds)

            # Chọn câu có điểm cao nhất
            if len(completed_scores) > 0:
                max_idx = completed_scores.index(max(completed_scores))
                return completed_seqs[max_idx].tolist()
            else:
                return seqs[0].tolist()
