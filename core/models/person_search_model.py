from functools import partial
import torch
import torch.nn.functional as F
from torch import nn

from models.vit import VisionTransformer
from models.xbert import BertConfig, BertForMaskedLM


class ALBEF(nn.Module):
    def __init__(
        self,
        text_encoder=None,
        tokenizer=None,
        config=None,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        embed_dim = config["embed_dim"]  # out_features: 256

        ###  Text encoder  ###
        bert_config = BertConfig.from_json_file(config["bert_config"])
        self.text_encoder = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)
        # in_features: 768 (must be a multiple of the number of attention heads (12))
        self.text_width = self.text_encoder.config.hidden_size
        self.text_proj = nn.Linear(self.text_width, embed_dim)  # 768 → 256

        ###  Vision encoder  ###
        vision_width = config["vision_width"]
        self.visual_encoder = VisionTransformer(
            img_size=config["image_res"],
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
        self.vision_proj = nn.Linear(vision_width, embed_dim)  # 768 → 256

        ###  Momentum models  ###
        self.text_encoder_m = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)
        self.text_proj_m = nn.Linear(self.text_width, embed_dim)
        self.visual_encoder_m = VisionTransformer(
            img_size=config["image_res"],
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.model_pairs = [
            [self.visual_encoder, self.visual_encoder_m],
            [self.vision_proj, self.vision_proj_m],
            [self.text_encoder, self.text_encoder_m],
            [self.text_proj, self.text_proj_m],
        ]
        self.copy_params()

        ###  Create the queue  ###
        self.queue_size = config["queue_size"]
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))  #  tensor([256, 65536])
        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))  #  tensor([256, 65536])
        self.register_buffer("idx_queue", torch.full((1, self.queue_size), -100))  # tensor([1, 65536]) filled with -100
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  # tensor([1])  |  Queue pointer
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

    def forward(self, image1, image2, text1, text2, alpha, idx, replace):
        ###  Extract text features  ###
        text_output = self.text_encoder.bert(
            text2.input_ids, attention_mask=text2.attention_mask, return_dict=True, mode="text"
        )
        text_embeds = text_output.last_hidden_state  # tensor([batch, seq_length, hidden_size])
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

        ###  Extract image features  ###
        image_embeds = self.visual_encoder(image1)  # tensor([batch, num_tokens, vision_width])
        # num_tokens = (img_size / patch_size)^2 + 1 (CLS token)  ||  (384 / 16)^2 + 1 = 577
        image_attn_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image1.device)
        # image_attn_mask = tensor([batch, num_tokens])
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

        # Contrastive loss
        idx = idx.view(-1, 1)  # reshape from [batch_size] to [batch_size, 1]
        idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()], dim=1)
        pos_idx = torch.eq(idx, idx_all).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

        with torch.no_grad():
            self._momentum_update()

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # Copy weights from online to momentum
                param_m.requires_grad = False  # Prevents gradient updates in momentum model

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1.0 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, idx):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)
        idxs = concat_all_gather(idx)
        batch_size = image_feats.shape[0]
        ptr = int(self.queue_ptr)
        # replace the keys at ptr (dequeue and enqueue)
        empty = self.image_queue.size(1) - ptr
        if batch_size <= empty:
            self.image_queue[:, ptr : ptr + batch_size] = image_feats.T
            self.text_queue[:, ptr : ptr + batch_size] = text_feats.T
            self.idx_queue[:, ptr : ptr + batch_size] = idxs.T
        else:
            self.image_queue[:, ptr:] = image_feats[:empty].T
            self.text_queue[:, ptr:] = text_feats[:empty].T
            self.idx_queue[:, ptr:] = idxs[:empty].T
            self.image_queue[:, : batch_size - empty] = image_feats[empty:].T
            self.text_queue[:, : batch_size - empty] = text_feats[empty:].T
            self.idx_queue[:, : batch_size - empty] = idxs[empty:].T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        self.queue_ptr[0] = ptr

    def mask(self, input_ids, vocab_size, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False
        if targets is not None:
            targets[~masked_indices] = -100  # We only compute loss on masked tokens
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id
        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(input_ids.device)
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        if targets is not None:
            return input_ids, targets
        else:
            return input_ids

    def mrtd_mask_modeling(self, mrtd_input_ids, ori_input_ids, attention_mask, weights):
        bs = mrtd_input_ids.size(0)
        weights = weights.view(-1, weights.size(-1))
        pred = torch.multinomial(weights, 1).view(bs, -1)
        pred[:, 0] = self.tokenizer.cls_token_id
        # pad_token_id is 0
        mrtd_input_ids = pred * attention_mask
        mrtd_labels = (pred != ori_input_ids) * attention_mask
        mrtd_labels[mrtd_input_ids == self.tokenizer.pad_token_id] = -100
        mrtd_labels[mrtd_input_ids == self.tokenizer.cls_token_id] = -100
        return mrtd_input_ids, mrtd_labels


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    If not in distributed mode, it simply returns the input tensor.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        tensors_gather = [torch.ones_like(tensor) for _ in range(world_size)]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
        return torch.cat(tensors_gather, dim=0)

    # If not in distributed mode, return the tensor as is
    return tensor
