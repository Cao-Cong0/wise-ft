import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import requests
import pandas as pd
from io import BytesIO

class MyDataset(Dataset):
    def __init__(self, preprocess_fn, location, batch_size, classnames=None, is_test=False):
        print(f'Loading csv data from {location}.')
        self.is_test = is_test
        self.data = pd.read_csv(location, header=0)
        raw_captions = self.data['title'].tolist()
        raw_image_urls = self.data['filepath'].tolist()
        self.captions = self.data['title'].tolist()
        self.image_urls = self.data['filepath'].tolist()
        self.preprocess_fn = preprocess_fn
        self.loader_args = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 1}
        print('Done loading data.')
        self.loader_args_train = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 1}
        self.loader_args_test = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 1}
        self.classnames = classnames  # Store the classnames
        # if self.is_test:
        #     self.loader_args = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 1}
        # else:
        #     self.loader_args = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 1}
            
        # Filter the data
        self.captions = []
        self.image_urls = []
        for caption, url in zip(raw_captions, raw_image_urls):
            if self.is_valid_url(url):  # Check if the URL is valid
                self.captions.append(caption)
                self.image_urls.append(url)
                # print(f"{caption}, valid URL: {url}")
            else:
                print(f"Ignoring invalid URL: {url}")
                
    @property
    def train_loader(self):
        return DataLoader(self, **self.loader_args_train)

    @property
    def test_loader(self):
        return DataLoader(self, **self.loader_args_test)
    
    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image_url = self.image_urls[idx]
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        image = self.preprocess_fn(image)

        class_to_idx = {'porn': 0, 'political': 1, 'sexy': 2, 'cigar-alcohol': 3, 'insult': 4}
        return {'images': image, 'labels': class_to_idx[self.captions[idx]]}

    def is_valid_url(self, url):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                # print(f"Failed to download image at {url}")
                return False
            if not response.headers['Content-Type'].startswith('image'):
                # print(f"URL does not point to an image: {url}")
                return False
            image = Image.open(BytesIO(response.content))
            return True
        except (requests.exceptions.RequestException, IOError, SyntaxError) as e:
            # print(f"Cannot open or identify image at {url}.")
            return False
        return True