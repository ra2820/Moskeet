
from io import BytesIO
import numpy as np
from numpy.core.arrayprint import DatetimeFormat
from numpy.testing._private.utils import assert_equal 
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler

import unittest

from dataloader_mbn import MBN

class Test_MBNDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mbn_dataset = MBN(csv_file_path='C:\\Users\\roopa\\OneDrive\\Desktop\\MOSKEET\\Entire_data_final\\combined_split_csv.csv')

        df = pd.read_csv('C:\\Users\\roopa\\OneDrive\\Desktop\\MOSKEET\\Entire_data_final\\combined_split_csv.csv')

        index = df.index
        condition_train = df["Set"] == "train"
        condition_test = df["Set"] == "test"
        condition_val = df["Set"] == "val"
        train_indices = index[condition_train]
        test_indices = index[condition_test]
        val_indices = index[condition_val]


        cls.train_sampler = SubsetRandomSampler(train_indices)
        cls.test_sampler = SubsetRandomSampler(test_indices)
        cls.valid_sampler = SubsetRandomSampler(val_indices)


        cls.dataloader_train = DataLoader(cls.mbn_dataset, batch_size=100,sampler = cls.train_sampler,num_workers=0)
        cls.dataloader_test = DataLoader(cls.mbn_dataset, batch_size=100,sampler = cls.test_sampler, num_workers=0)
        cls.dataloader_val = DataLoader(cls.mbn_dataset, batch_size=100,sampler = cls.valid_sampler, num_workers=0)

        cls.samples_train = next(iter(cls.dataloader_train))
        cls.samples_test = next(iter(cls.dataloader_test))
        cls.samples_val = next(iter(cls.dataloader_val))
    
    def test_number_samples(self):
        self.assertEqual(len(Test_MBNDataset.dataloader_train), int(round(len(Test_MBNDataset.train_sampler)/(Test_MBNDataset.dataloader_train.batch_size))))
        self.assertEqual(len(Test_MBNDataset.dataloader_test), int(round(len(Test_MBNDataset.test_sampler)/(Test_MBNDataset.dataloader_test.batch_size))))
        self.assertEqual(len(Test_MBNDataset.dataloader_val), int(round(len(Test_MBNDataset.valid_sampler)/(Test_MBNDataset.dataloader_val.batch_size))))
        self.assertEqual(len(Test_MBNDataset.train_sampler) + len(Test_MBNDataset.test_sampler) + len(Test_MBNDataset.valid_sampler), len(Test_MBNDataset.mbn_dataset))

    
    def test_tensor(self):
        sample_shape_train = Test_MBNDataset.samples_train['mbn_tensor'].shape
        sample_shape_test = Test_MBNDataset.samples_test['mbn_tensor'].shape
        sample_shape_val = Test_MBNDataset.samples_val['mbn_tensor'].shape

        sample_type_train = type(Test_MBNDataset.samples_train['mbn_tensor'])
        sample_type_test = type(Test_MBNDataset.samples_test['mbn_tensor'])
        sample_type_val = type(Test_MBNDataset.samples_val['mbn_tensor'])

        self.assertEqual(sample_shape_train[0], Test_MBNDataset.dataloader_train.batch_size)
        self.assertEqual(sample_shape_test[0], Test_MBNDataset.dataloader_test.batch_size)
        self.assertEqual(sample_shape_val[0], Test_MBNDataset.dataloader_val.batch_size)

        self.assertEqual(sample_type_train, torch.Tensor)
        self.assertEqual(sample_type_test, torch.Tensor)
        self.assertEqual(sample_type_val, torch.Tensor)
    
    def test_type_samples(self): 
        for idx,sample in enumerate(Test_MBNDataset.dataloader_train): 
    
            if idx == (len(Test_MBNDataset.train_sampler) // Test_MBNDataset.dataloader_train.batch_size): # check all samples from train
                self.assertEqual(sample['set'], ['train']*(len(Test_MBNDataset.train_sampler)% Test_MBNDataset.dataloader_train.batch_size))
            else:
                self.assertEqual(sample['set'], ['train']*Test_MBNDataset.dataloader_train.batch_size)
        
        for idx,sample in enumerate(Test_MBNDataset.dataloader_test): 
    
            if idx == (len(Test_MBNDataset.test_sampler) // Test_MBNDataset.dataloader_test.batch_size): 
                self.assertEqual(sample['set'], ['test']*(len(Test_MBNDataset.test_sampler)% Test_MBNDataset.dataloader_test.batch_size))
            else:
                self.assertEqual(sample['set'], ['test']*Test_MBNDataset.dataloader_test.batch_size)
        
        for idx,sample in enumerate(Test_MBNDataset.dataloader_val): 
    
            if idx == (len(Test_MBNDataset.valid_sampler) // Test_MBNDataset.dataloader_val.batch_size): 
                self.assertEqual(sample['set'], ['val']*(len(Test_MBNDataset.valid_sampler)% Test_MBNDataset.dataloader_val.batch_size))
            else:
                self.assertEqual(sample['set'], ['val']*Test_MBNDataset.dataloader_val.batch_size)
