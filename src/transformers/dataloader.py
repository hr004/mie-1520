import random
import torch
import torchtext
from torchtext import data, datasets
from torchtext import vocab
from argparse import ArgumentParser


def get_imdb_data(args):
    """
    Downloads the IMDB dataset and prepares it for training and validation.

    Args:
        args: An object containing the necessary arguments for data preparation.

    Returns:
        train_it: The training data iterator.
        valid_it: The validation data iterator.
    """

    # get txt and label field
    txt_field, label_field = define_fields(args)
    datasets.IMDB.download("./")
    train_ds, test_ds = datasets.IMDB.splits(txt_field, label_field)
    # no validation
    txt_field.build_vocab(
        train_ds,
        test_ds,
        max_size=args.vocab_size - 2,  # default for <unk> and <pad>
        min_freq=1,  # default
        vectors=get_vectors(args),
        vectors_cache=None,
    )
    label_field.build_vocab(train_ds)
    train_it, valid_it = batchify(args, train_ds, test_ds)
    return train_it, valid_it


def get_data(args):
    """
    Get the train and validation data iterators.

    Args:
        args: The arguments containing the configuration for data loading.

    Returns:
        train_it: The iterator for the training data.
        valid_it: The iterator for the validation data.
    """
    txt_field, label_field = define_fields(args)
    # train and validation fields
    train_val_fields = [("text"), txt_field, ("label"), label_field]
    dataset = load_data(args, train_val_fields, txt_field, label_field)
    trainds, valds, testds = split_data(args, dataset)
    # build vocabulary
    # consider train and valds to create vocabulary
    txt_field.build_vocab(
        trainds,
        valds,
        max_size=args.vocab_size - 2,  # default for <unk> and <pad>
        min_freq=1,  # default
        vectors=get_vectors(args),
        vectors_cache=None,
    )
    label_field.build_vocab(trainds)
    train_it, valid_it = batchify(args, trainds, valds)
    return train_it, valid_it  # len(txt_field.vocab), txt_field.vocab.vectors


def define_fields(args):
    """
    Define and return the text and label fields for data processing.

    Args:
        args: A namespace object containing the arguments.

    Returns:
        txt_field: A Field object for text-related data processing.
        label_field: A Field object for label-related data processing.
    """

    # text related Field
    txt_field = data.Field(
        sequential=True,
        use_vocab=True,  # set true to create build vocabulary
        fix_length=args.max_seq_len,  # default
        dtype=torch.long,
        include_lengths=False,
        batch_first=True,
        stop_words=None,
        is_target=False,
        pad_token="<pad>",
        unk_token="<unk>",
    )
    # label related Field
    label_field = data.Field(
        sequential=False,
        use_vocab=True,
        pad_token=None,
        unk_token=None,
    )
    return txt_field, label_field


def load_data(args, train_val_fields, text_field, label_field):
    # create a tabular dataset
    dataset = data.TabularDataset(
        path=args.train_path,
        format=args.data_format,
        fields=train_val_fields,
        skip_header=True,
        filter_pred=None,  # filter data by a specific  class "positive"
        csv_reader_params={},
    )
    return dataset


def split_data(args, dataset):
    trainds, valds, testds = dataset.split(
        split_ratio=[0.7, 0.1, 0.2], random_state=random.getstate()
    )
    return trainds, valds, testds


def get_vectors(args):
    if args.pretrained_vec:
        print("Loading pretrained vectors")
        # this can be a list of pretrained vectors
        # glove = torchtext.vocab.Vectors('glove.6B.50d.txt', cache='./tmp')
        glove = torchtext.vocab.GloVe(name="840B", dim=300, cache="./tmp")
        return glove
    else:
        return


def batchify(args, trainds, valds, testds=None):
    # batching
    train_it, valid_it = data.BucketIterator.splits(
        (trainds, valds),
        batch_sizes=(args.batch_size, args.batch_size),
        device=args.device,
        sort_key=lambda x: len(x.text),
        sort_within_batch=False,
        repeat=False,
    )
    return train_it, valid_it


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--builtin_data", type=bool, default=True)
    parser.add_argument("--pretrained_vec", type=bool, default=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    data = get_data(args)
