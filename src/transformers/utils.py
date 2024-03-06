import os
import torch
from termcolor import cprint, colored
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import torch


def compute_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average="micro")
    recall = recall_score(y_true, y_pred, average="micro")
    f1score = f1_score(y_true, y_pred, average="micro")
    accuracy = accuracy_score(y_true, y_pred)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1score,
        "accuracy": accuracy,
    }


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
