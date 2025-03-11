import json
import os
import numpy as np
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset
from collections import defaultdict

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class ps_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30, weak_pos_pair_probability=0.1):
        anns = []
        for f in ann_file:
            anns += json.load(open(f, "r"))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.weak_pos_pair_probability = weak_pos_pair_probability  # 待修改
        self.person2image = defaultdict(list)
        self.person2text = defaultdict(list)
        person_id2idx = {}
        n = 0
        self.pairs = []
        for ann in anns:
            person_id = ann["id"]
            if person_id not in person_id2idx.keys():
                person_id2idx[person_id] = n
                n += 1
            person_idx = person_id2idx[person_id]
            self.person2image[person_idx].append(ann["file_path"])
            for cap in ann["captions"]:
                self.pairs.append((ann["file_path"], cap, person_idx))
                self.person2text[person_idx].append(cap)

    def __len__(self):
        return len(self.pairs)

    def augment(self, caption, person):
        caption_aug = caption
        if np.random.random() < self.weak_pos_pair_probability:
            caption_aug = np.random.choice(self.person2text[person], 1).item()
        if caption_aug == caption:
            replace = 0
        else:
            replace = 1
        return caption_aug, replace

    def __getitem__(self, index):
        image_path, caption, person = self.pairs[index]
        caption_aug, replace = self.augment(caption, person)
        image_path = os.path.join(self.image_root, image_path)
        image = Image.open(image_path).convert("RGB")
        image1 = self.transform(image)
        image2 = self.transform(image)
        caption1 = pre_caption(caption, self.max_words)
        caption2 = pre_caption(caption_aug, self.max_words)
        return image1, image2, caption1, caption2, person, replace
