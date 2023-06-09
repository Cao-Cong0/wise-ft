import os
import numpy as np
import torch
from src.models.eval import evaluate
from src.models.finetune import finetune
from src.models.modeling import ClassificationHead, ImageEncoder, ImageClassifier
from src.models.utils import fisher_load
from src.models.zeroshot import get_zeroshot_classifier
from src.args import parse_arguments
from src.models.test import test

if __name__ == '__main__':
    args = parse_arguments()
    test(args)

