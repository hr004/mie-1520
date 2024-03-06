import argparse

parser = argparse.ArgumentParser(description="dataloader")
# data
data = parser.add_argument_group("Training Data Options")
data.add_argument(
    "--train_path", type=str, default="data/all.csv", help="training file path"
)
data.add_argument("--data_format", type=str, default="csv", help="training data format")
data.add_argument("--val_path", type=str, default=None, help="validation file path")
data.add_argument(
    "--builtin_data", default=True, action="store_true", help="whether to use IMDB data"
)
data.add_argument(
    "--pretrained_vec",
    default=False,
    action="store_true",
    help="use pretrained vocabulary",
)
data.add_argument("--vocab_size", type=int, default=50002, help="Vocabulary size")

# learning
learn = parser.add_argument_group("Learning Options")
learn.add_argument(
    "--batch_size", type=int, default=64, help="use pretrained vocabulary"
)
learn.add_argument("--epochs", type=int, default=10, help="number of training epochs")
learn.add_argument("--optimizer", default="AdamW", help="Optimizer name")
learn.add_argument("--lr", default=0.001, help="Learning rate")
learn.add_argument("--lr_warmup", default=10000)
learn.add_argument("--dynamic_lr", default=False, help="dynamic learning")
learn.add_argument("--mode", default=["non", "static"], help="define mode for weights")
learn.add_argument(
    "--max_norm", default=300, help="Norm cutoff to prevent gradient explosion"
)
learn.add_argument(
    "--milestones",
    nargs="+",
    default=[5, 10, 15],
    help="List of epoch indices. Must be increasing",
)
learn.add_argument(
    "--decay_factor",
    default=0.5,
    type=float,
    help="Decay factor to reduce learning rate",
)
learn.add_argument(
    "--gradient_clipping", default=1.0, type=float, help="Gradient clipping to use"
)
learn.add_argument("--tb_dir", default="./runs")
# model
transformer = parser.add_argument_group("Model related options")
transformer.add_argument(
    "--embedding_dim", type=int, default=128, help="Embedding dimension"
)
transformer.add_argument(
    "--output_dim", type=int, default=2, help="Output dimension or number of classes"
)
# transformer.add_argument("--bidirection", default=True, action="store_true", help="Use Bi-LSTM")
transformer.add_argument("--depth", type=int, default=1, help="Transformer layers")
transformer.add_argument(
    "--num_heads", type=int, default=2, help="Number of attention heads"
)
transformer.add_argument(
    "--dropout", type=float, default=0.0, help="dropout probability"
)
# transformer.add_argument("--num_tokens", type=int, default=50002, help="Vocabulary size")
transformer.add_argument(
    "--max_pool",
    type=bool,
    default=True,
    help="Use max pool; if not true mean pool will be used",
)
transformer.add_argument(
    "--max_seq_len", type=int, default=512, help="Maximum sequence length"
)
# transformer.add_argument("--n_hidden", type=int, default=50, help="hidden dimension")
# transformer.add_argument("--num_layers", type=int, default=2, help="number of hidden layers")
# device
device = parser.add_argument_group("Device Options")
device.add_argument("--cuda", default=True, action="store_true", help="Use GPU")
# experiment
experiment = parser.add_argument_group("Experimental Options")
experiment.add_argument(
    "--verbose",
    dest="verbose",
    action="store_true",
    default=False,
    help="Verbosity for debugging",
)
experiment.add_argument(
    "--continue_from",
    default="",
    help="indicate path of saved model to continue training",
)
experiment.add_argument(
    "--checkpoint", dest="checkpoint", default=True, help="Checkpoint to save model"
)
experiment.add_argument(
    "--checkpoint_per_batch",
    default=1000,
    type=int,
    help="Save checkpoint per batch, 0 means never save",
)
experiment.add_argument(
    "--save_folder",
    default="saved_models/Attntransformer",
    help="Location of the save models",
)
experiment.add_argument(
    "--log_config",
    default=True,
    action="store_true",
    help="Store experiment configuration",
)
experiment.add_argument("--log_interval", default=10, help="log interval")
experiment.add_argument("--val_interval", default=200, help="val interval")
experiment.add_argument(
    "--evaluate_within_epoch",
    default=True,
    help="Whether to call evaluation func within an epoch",
)
experiment.add_argument(
    "--save_interval", default=5, help="Number of epoch to wait to save the model"
)

args = parser.parse_args()
