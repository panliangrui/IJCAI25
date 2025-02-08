import os

import torch
import glob
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from random import shuffle
from torch.utils.data import Dataset


class base(Dataset):
    def __init__(self, data, split_path,batch_size, sheet_name="train",need_shuffle=False):
        super().__init__()
        self.data=data
        self.sheet_name=sheet_name
        self.batch_size=batch_size
        self.need_shuffle=need_shuffle

        # 从csv文件中提取某一个sheet表中的file_name列和label列数据
        file_labels_df = pd.read_excel(split_path, sheet_name=sheet_name)
        train_slide_names = file_labels_df['file_name'].tolist()
        pt_files = [data + "/" + slide_name + ".pt" for slide_name in train_slide_names]
        self.pt_files = pt_files
        self.pt_label = file_labels_df['label'].tolist()


        print("finishing")
    def init_train_step(self,train_num):
        if train_num % self.batch_size == 0:
            self.train_step =train_num // self.batch_size
        else:
            self.train_step = (train_num // self.batch_size) + 1

    def get_random_indexs(self):
        indexs = list(range(len(self.x)))
        if self.need_shuffle:
            shuffle(indexs)

        new_random_index=[]
        start=0
        for i in range(self.train_step):
            end=start+self.batch_size

            if end<len(indexs):
                new_random_index.append(indexs[start:end])
            else:
                new_random_index.append(indexs[start:])
                break

            start+=self.batch_size
        return new_random_index

    def __len__(self):
        return len(self.pt_label)



class myDataset(base):
    def __init__(self, data, split_path, sheet_name="train"):
        super().__init__(data, split_path, sheet_name)
        pass

    def __getitem__(self, index):
        path = self.pt_files[index]
        data = torch.load(path, map_location="cpu")
        return data, self.pt_label[index]


if __name__ == "__main__":
    repre_dir = "D:\\HNSZL Train\\represention\\r50_level0_224\\stage3"
    test_split_xlsl = "D:\\project\\laten_mamba_main\\data\\LUNG\\luad_lusc_normal_Lung_train_file_labels.xlsx"
    test_sheet = "train"

    dataset = myDataset(data=repre_dir,
                        split_path=test_split_xlsl,
                        sheet_name=test_sheet)
    for i in range(len(dataset)):
        print(dataset.__getitem__(i))
    # dataset = EmbededFeatsDataset('/your/path/to/CAMELYON16/', mode='test')
