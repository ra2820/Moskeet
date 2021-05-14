from io import BytesIO
import numpy as np
from numpy.core.arrayprint import DatetimeFormat 
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler



class MBN(Dataset):
    """MBN dataset."""

    def __init__(self, csv_file_path):
        """
        Args:
            csv_file (string): Path to the csv file with train/val/test splits.
            
        """
        self.mbn_files_data = pd.read_csv(csv_file_path)
        

    def __len__(self):
        return len(self.mbn_files_data)

    def __getitem__(self, idx):

        file_path = self.mbn_files_data['Path'][idx]
        data = open(str(file_path), "rb").read()
        mbn_numpy = np.frombuffer(data, dtype=np.uint8) #dtype = np.uint16 or float
        mbn_tensor = torch.from_numpy(np.array(mbn_numpy))
        species = self.mbn_files_data['Species'][idx]
        set = self.mbn_files_data['Set'][idx]
        


        sample = {'mbn_tensor': mbn_tensor, 'species': species, 'set':set}

        return sample

mbn_dataset = MBN(csv_file_path='C:\\Users\\roopa\\OneDrive\\Desktop\\MOSKEET\\Entire_data_final\\combined_split_csv.csv')

sample = mbn_dataset[20000]
print(sample['mbn_tensor'])
print(sample['species'])

df = pd.read_csv('C:\\Users\\roopa\\OneDrive\\Desktop\\MOSKEET\\Entire_data_final\\combined_split_csv.csv')

index = df.index
condition_train = df["Set"] == "train"
condition_test = df["Set"] == "test"
condition_val = df["Set"] == "val"
train_indices = index[condition_train]
test_indices = index[condition_train]
val_indices = index[condition_train]


train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)
valid_sampler = SubsetRandomSampler(val_indices)


dataloader_train = DataLoader(mbn_dataset, batch_size=100,sampler = train_sampler,num_workers=0)
dataloader_test = DataLoader(mbn_dataset, batch_size=100,sampler = test_sampler, num_workers=0)
dataloader_val = DataLoader(mbn_dataset, batch_size=100,sampler = valid_sampler, num_workers=0)




