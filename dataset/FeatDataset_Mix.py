import torch
from dataset.BaseDataset import base
import os
class FeatDataset_Mix(base):
    def __init__(self,
                 data,
                 split_path,
                 sheet_name,
                 batch_size,
                 need_shuffle=True,
                 ):
        super().__init__(data, split_path,batch_size, sheet_name,need_shuffle)

        self.load_data()
        self.init_train_step(len(self.x))

    def load_data(self,):
        self.x,self.y=[],[]
        for i,(pt,la) in enumerate(zip(self.pt_files,self.pt_label)):
            if not os.path.exists(pt):
                continue
            instances = torch.load(pt,map_location="cpu")

            if i < 150:
                instances =  instances.to("cuda")

            self.x.append(instances)
            self.y.append(la)


    def get_batch(self,index):
        x,y, batch = [], [], []

        for c, i in enumerate(index):
            instances = self.x[i].to("cuda")
            instance_y=self.pt_label[i]
            bag_batch = torch.ones(size=(len(instances),), dtype=torch.int64,device="cuda") * (c % self.batch_size)

            x.append(instances)
            y.append(instance_y)
            batch.append(bag_batch)

        x=torch.cat(x)
        y=torch.tensor(y).to("cuda")
        batch=torch.cat(batch)

        data={}
        data["x"]=x
        data["y"]=y
        data["batch"]=batch
        return data
