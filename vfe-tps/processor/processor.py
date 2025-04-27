import logging


def do_train(
    start_epoch,
    args,
    model,
    train_loader,
):
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("IRRA.train")
    logger.info("start training")

    args.training = False

    # train
    # TEST 1 epoch
    num_epoch = 1
    for epoch in range(start_epoch, num_epoch + 1):
        model.train()
        model.clear_dic()

        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            ret = model(batch)
            # total_loss = sum([v for k, v in ret.items() if "loss" in k])
