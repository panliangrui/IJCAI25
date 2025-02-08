
from dataset.FeatDataset_Mix import FeatDataset_Mix

def load_dataset(
                 data,
                 split_path,
                 sheet_name,
                 batch_size,
                 need_shuffle):
    return FeatDataset_Mix(
        data,
        split_path,
        sheet_name,
        batch_size,
        need_shuffle)


def load_dataloader(train,args):
    if train=="train":
        return load_dataset(
                             args.train_data,
                             args.excel_path,
                             args.train_sheet,
                             args.train_bs,
                             True)
    elif train=="test":
        return load_dataset(
                             args.test_data,
                             args.excel_path,
                             args.test_sheet,
                             args.test_bs,
                             False)