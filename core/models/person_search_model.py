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
        self.temp = nn.Parameter(torch.ones([]) * config["temp"])
        self.momentum = config["momentum"]

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
        self.itm_head = nn.Linear(self.text_width, 2)
        self.prd_head = nn.Linear(self.text_width, 2)
        self.mrtd_head = nn.Linear(self.text_width, 2)

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
            text_output_m = self.text_encoder_m.bert(
                text2.input_ids, attention_mask=text2.attention_mask, return_dict=True, mode="text"
            )
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1)
            text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

            image_embeds_m = self.visual_encoder_m(image2)
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)
            image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)

            sim_i2t_m = image_feat_m @ text_feat_all / self.temp
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp
            sim_i2i_m = image_feat_m @ image_feat_all / self.temp
            sim_t2t_m = text_feat_m @ text_feat_all / self.temp

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets
            sim_i2i_targets = alpha * F.softmax(sim_i2i_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2t_targets = alpha * F.softmax(sim_t2t_m, dim=1) + (1 - alpha) * sim_targets

        sim_i2t = image_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ image_feat_all / self.temp
        sim_i2i = image_feat @ image_feat_all / self.temp
        sim_t2t = text_feat @ text_feat_all / self.temp

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
        loss_i2i = -torch.sum(F.log_softmax(sim_i2i, dim=1) * sim_i2i_targets, dim=1).mean()
        loss_t2t = -torch.sum(F.log_softmax(sim_t2t, dim=1) * sim_t2t_targets, dim=1).mean()
        loss_cl = (loss_i2t + loss_t2i + loss_i2i + loss_t2t) / 4

        self._dequeue_and_enqueue(image_feat_m, text_feat_m, idx)

        # Relation-aware Learning: Probabilistic Image-Text Matching + Positive Relation Detection
        # Probabilistic Image-Text Matching
        # forward the positive image-text pairs
        output_pos = self.text_encoder.bert(
            encoder_embeds=text_embeds,
            attention_mask=text2.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attn_mask,
            return_dict=True,
            mode="fusion",
        )  # tensor([batch, seq_length, hidden_size])

        with torch.no_grad():
            bs = image1.size(0)
            weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1)  # tensor([batch, batch])
            weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1)
            mask = torch.eq(idx, idx.T)  # tensor([batch, batch]) of Bool
            weights_i2t.masked_fill_(mask, 0)
            weights_t2i.masked_fill_(mask, 0)

        # Select a negative image for each text
        image_neg_idx = torch.multinomial(weights_t2i, 1).flatten()
        image_embeds_neg = image_embeds[image_neg_idx]
        # select a negative text for each image
        text_neg_idx = torch.multinomial(weights_i2t, 1).flatten()
        text_embeds_neg = text_embeds[text_neg_idx]
        text_attn_neg = text2.attention_mask[text_neg_idx]
        # forward the negative image-text pairs
        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
        text_attn_all = torch.cat([text2.attention_mask, text_attn_neg], dim=0)
        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_attn_all = torch.cat([image_attn_mask, image_attn_mask], dim=0)
        output_neg_cross = self.text_encoder.bert(
            encoder_embeds=text_embeds_all,
            attention_mask=text_attn_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_attn_all,
            return_dict=True,
            mode="fusion",
        )
        vl_embeddings = torch.cat(
            [output_pos.last_hidden_state[:, 0, :], output_neg_cross.last_hidden_state[:, 0, :]], dim=0
        )
        vl_output = self.itm_head(vl_embeddings)
        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)], dim=0).to(
            image1.device
        )
        loss_pitm = F.cross_entropy(vl_output, itm_labels)

        # Positive Relation Detection
        prd_output = self.prd_head(output_pos.last_hidden_state[:, 0, :])
        loss_prd = F.cross_entropy(prd_output, replace)
        print("loss: ", loss_cl, loss_pitm, loss_prd)
        return loss_cl

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
