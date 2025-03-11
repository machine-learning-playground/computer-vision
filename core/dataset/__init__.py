from torch.utils.data import DataLoader


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
