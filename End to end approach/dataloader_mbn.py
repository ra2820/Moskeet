from io import BytesIO
import numpy as np
from numpy.core.arrayprint import DatetimeFormat
from numpy.testing._private.utils import assert_equal 
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
        with open(str(file_path), "rb") as f: 
            data = f.read()
       
            mbn_numpy = np.frombuffer(data, dtype=np.uint8) #dtype = np.uint16 or float
            mbn_tensor = torch.from_numpy(np.array(mbn_numpy))
            species = self.mbn_files_data['Species'][idx]
            set = self.mbn_files_data['Set'][idx]
            sample = {'mbn_tensor': mbn_tensor, 'species': species, 'set':set}
            return sample



