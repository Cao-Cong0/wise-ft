import csv
import requests
from PIL import Image
from io import BytesIO
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

class MyDataset(Dataset):
    def __init__(self, preprocess_fn, location, batch_size, classnames=None, is_test=False):
        print(f'Loading csv data from {location}.')
        self.is_test = is_test
        self.data = pd.read_csv(location, header=0)
        self.captions = self.data['title'].tolist()
        self.image_urls = self.data['filepath'].tolist()
        self.preprocess_fn = preprocess_fn
        self.loader_args_train = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 1}
        self.loader_args_test = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 1}
        self.classnames = classnames

    @property
    def train_loader(self):
        return DataLoader(self, **self.loader_args_train, collate_fn=collate_fn)

    @property
    def test_loader(self):
        return DataLoader(self, **self.loader_args_test, collate_fn=collate_fn)


    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image_url = self.image_urls[idx]
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            image = self.preprocess_fn(image)

            class_to_idx = {'porn': 0, 'political': 1, 'cigar-alcohol': 2, 'insult': 3, 'violent': 4, 'normal': 5}
            return {'images': image, 'labels': class_to_idx[self.captions[idx]]}
        except:
            # Raise StopIteration here to skip this sample during training/testing
            #raise StopIteration()return {'images': image, 'labels': class_to_idx[self.captions[idx]]}
            return None
