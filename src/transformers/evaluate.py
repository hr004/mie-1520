import torch
from src.transformers.utils import compute_metrics


def evaluate(args, model, loss_fn, data_iterator):
    """
    Evaluate the performance of a model on a given dataset.

    Args:
        args (argparse.Namespace): The command-line arguments.
        model: The transformer model to evaluate.
        loss_fn: The loss function used for evaluation.
        data_iterator: The iterator over the evaluation dataset.

    Returns:
        Tuple[float, float]: The average loss and accuracy over the evaluation dataset.
    """

    model.eval()
    val_acc = 0
    losses = 0.0
    with torch.no_grad():
        for idx, batch in enumerate(data_iterator):
            x = batch.text.to(args.device)
            y = batch.label.to(args.device)
            out = model(x)
            loss = loss_fn(out, y).data
            losses += loss.item()
            y_pred = out.argmax(1).cpu().numpy()
            metrics = compute_metrics(y.cpu().numpy(), y_pred)
            val_acc += metrics["accuracy"]
    epoch_val_acc = val_acc / len(data_iterator)
    epoch_val_loss = losses / len(data_iterator)
    return epoch_val_loss, epoch_val_acc
