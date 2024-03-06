import torch
import torch.nn as nn
from src.transformers.evaluate import evaluate
from src.transformers.utils import compute_metrics


def train(
    args,
    model,
    epoch,
    train_dl,
    loss_fn,
    optimizer,
    val_dl=None,
    scheduler=None,
):
    train_acc = 0
    model.train()
    losses = 0.0

    for i_batch, batch in enumerate(train_dl):
        x = batch.text.to(args.device)
        y = batch.label.to(args.device)
        out = model(x)
        loss = loss_fn(out, y)
        model.zero_grad()
        loss.backward()
        losses += loss.item()
        # clip gradients
        # - If the total gradient vector has a length > 1, we clip it back down to 1.
        if args.gradient_clipping > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)

        optimizer.step()
        scheduler.step()

        if args.cuda and args.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        y_pred = out.argmax(1).cpu().numpy()
        metrics = compute_metrics(y.cpu().numpy(), y_pred)
        train_acc += metrics["accuracy"]

    epoch_train_acc = train_acc / len(train_dl)
    epoch_train_loss = losses / len(train_dl)
    return epoch_train_loss, epoch_train_acc
