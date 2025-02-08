import numpy as np
import random
import torch
import os
import pandas as pd


def normalize_l2(x, axis=1):
    '''x.shape = (num_samples, feat_dim)'''
    x_norm = np.linalg.norm(x, axis=axis, keepdims=True)
    x = x / (x_norm + 1e-8)
    return x

def fix_random_seed(seed):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True

def get_subpath(dirpath,sort=False):
    path_list=os.listdir(dirpath)
    for i,path in enumerate(path_list):
        path_list[i]=os.path.normpath("%s/%s"%(dirpath,path))
    if sort:
        path_list.sort()
    return path_list
def get_subfolder_names(folder_path):
    #输出文件夹下不加后缀的文件夹名称
    subfolders = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path) and '.' not in item:
            subfolders.append(item)
    return subfolders


def save_dict_to_csv(dict_name,csv_file_path):
    max_length = max(len(lst) for lst in dict_name.values())
    for keys, values in dict_name.items():
        if len(values) >= max_length:
            continue
        else:
            for i in range(max_length - len(values)):
                values.append(0)
    df = pd.DataFrame(dict_name)
    df.to_csv(csv_file_path, index=False)

def Max_MIN_Tensor(input):
    max_value=torch.max(input)
    min_value=torch.min(input)
    return (input-min_value)/(max_value-min_value)

def load_to_device(data,device):
    for k,v in data.items():
        data[k]=v.to(device)
    return data

def random_shuffle(self,features,labels):
    # 将特征和标签打包成元组的列表
    data = list(zip(features, labels))
    # 随机打乱
    random.shuffle(data)
    # 解压缩得到打乱后的特征和标签
    shuffled_features, shuffled_labels = zip(*data)
    return torch.stack(shuffled_features), torch.stack(shuffled_labels)

def kmeans(tensor, k, max_iters=100, tolerance=1e-4):
    # 随机选择k个数据点作为初始聚类中心
    centroids = tensor[torch.randperm(tensor.size(0))[:k]]

    for i in range(max_iters):
        # 计算每个点到每个聚类中心的距离，并分配到最近的聚类中心
        distances = torch.cdist(tensor, centroids)
        cluster_assignments = torch.argmin(distances, dim=1)

        # 计算新的聚类中心
        new_centroids = torch.stack([tensor[cluster_assignments == j].mean(0) for j in range(k)])

        # 检查聚类中心是否收敛
        if torch.sum(torch.abs(new_centroids - centroids)) < tolerance:
            break

        centroids = new_centroids

    return centroids, cluster_assignments