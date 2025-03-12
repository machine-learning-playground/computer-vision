from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from dataset.ps_dataset import ps_train_dataset


def create_dataset(dataset, config):
    # mean for 3 RGB channels, std for 3 RGB channels
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    train_transform = transforms.Compose(
        [
            transforms.Resize((config["image_res"], config["image_res"]), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    if dataset == "ps":
        train_dataset = ps_train_dataset(
            config["train_file"],
            train_transform,
            config["train_image_root"],
            config["max_words"],
            config["weak_pos_pair_probability"],
        )
        return train_dataset


def create_loader(dataset, batch_size):
    # Train config
    sampler = None
    num_worker = 4
    collate_fn = None
    shuffle = sampler is None
    drop_last = True

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_worker,
        pin_memory=True,
        sampler=sampler,
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last=drop_last,
    )

    return loader
