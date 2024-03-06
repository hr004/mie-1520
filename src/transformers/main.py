import torch
import torch.nn as nn
import torch.optim as optim

from src.transformers.encoder import Encoder
from src.transformers.train import train
from src.transformers.evaluate import evaluate
from arguments import args
from utils import get_device
from src.transformers.dataloader import get_imdb_data


def main(args):
    args.device = get_device()
    train_iterator, valid_iterator = get_imdb_data(args)
    loss_fn = nn.CrossEntropyLoss()
    model = Encoder(
        emb=args.embedding_dim,
        heads=args.num_heads,
        depth=args.depth,
        seq_length=args.max_seq_len,
        vocab_size=args.vocab_size,
        num_classes=args.output_dim,
        max_pool=args.max_pool,
        dropout=args.dropout,
    )
    model = model.to(args.device)
    start_epoch = 1
    start_iter = 1

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.milestones,
        gamma=args.decay_factor,
        last_epoch=-1,
    )
    best_val_loss = float("inf")
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_train_loss, epoch_train_acc = train(
            args,
            model,
            epoch,
            train_iterator,
            loss_fn,
            optimizer,
            val_dl=valid_iterator,
            scheduler=scheduler,
        )

        epoch_val_loss, epoch_val_acc = evaluate(args, model, loss_fn, valid_iterator)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), args.save_folder + "/best_model.pth")

        print(
            f"Epoch: {epoch}/{args.epochs}, Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}"
        )


if __name__ == "__main__":
    print(args)
    main(args)
