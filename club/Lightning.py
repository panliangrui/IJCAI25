import numpy as np
import torch
import tqdm
from dataset.load_dataloader import load_dataloader
from club.Tensorboard_LG import tensorboard_lg
from club.checkpoint_LG import checkpoint_lg
from club.metric_LG import metric_lg
from club.stop_early_LG import stop_early_lg
from club.ranger import Ranger
from torch.nn import CrossEntropyLoss

class Lightning():
    def __init__(self,
                 args
                ):
        self.melg=metric_lg(metric_dir=args.metic_dir)
        self.stlg=stop_early_lg(metric=args.metric, patient=args.patient)
        self.tlg=tensorboard_lg(tensorboard_folder=args.tensorboard_dir)
        self.cklg=checkpoint_lg(metric=args.metric, checkpoint_dir=args.checkpoint_dir)
        self.args=args


    def train(self,
              model,
              ):

        optimizer = Ranger(model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        bag_loss_fn= CrossEntropyLoss()
        train_dataloder=load_dataloader("train",args=self.args)

        model.train()
        model.to(self.args.device)

        self.tlg.init_tensorbard(self.args.seed)

        for epoch in range(self.args.start_epoch,self.args.num_epochs):
            current_result = {"loss": 0,"acc":0}
            probs,ture=[],[]
            train_bar = tqdm.tqdm(train_dataloder.get_random_indexs(), desc="seed {} Training {}".format(self.args.seed,epoch))

            for train_index in train_bar:
                data=train_dataloder.get_batch(train_index)
                result = model(data)
                bag_loss=bag_loss_fn(result["probs"],data["y"])

                optimizer.zero_grad()
                bag_loss.backward()
                optimizer.step()

                bag_acc = self.melg.Accuracy(result["probs"], data["y"])
                current_result["acc"]+=bag_acc
                current_result["loss"]+=bag_loss.item()

                probs.append(result["probs"][:,-1].detach().cpu())
                ture.append(data["y"].cpu())

            self.tlg.refresh_log(epoch,recoder_dict=current_result,step=train_dataloder.train_step)

            # 保存模型
            self.cklg.save_epoch_checkpoint(
                epoch_metric=current_result[self.args.metric],
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                ckp_dir=self.cklg.checkpoint_dir,
                name=self.args.model_name,
                epoch_frq=self.args.epoch_frq,
                seed=self.args.seed)

            if self.stlg.stop(current_result[self.args.metric]):
                break
    def test_preditc(self,
                     model,
                     checkpoint_path=None,
                     ):
        if checkpoint_path is not None:
            self.cklg.load_checkpoint(model=model, path=checkpoint_path)
        test_dataloader = load_dataloader("test",args=self.args)
        model.to(self.args.device)
        model.eval()

        probs = []
        pred_labels = []
        true_labels = []
        test_bar = tqdm.tqdm(test_dataloader.get_random_indexs(), desc="Testing")
        with torch.no_grad():
            for test_index in test_bar:
                data= test_dataloader.get_batch(test_index)
                result= model(data)

                bag_probs=result["probs"]
                pred_label=torch.argmax(bag_probs)

                probs.extend(bag_probs.data.cpu().numpy())
                pred_labels.append(pred_label.item())
                true_labels.extend(data["y"])

        true_labels = torch.tensor(true_labels).numpy()
        probs = np.array(probs)
        pred_labels = np.array(pred_labels)
        return probs,true_labels,pred_labels

    def test(self,
             epoch,
             model,
             checkpoint_path=None,
             csv_path=None
             ):

        probs,true_labels ,pred_labels= self.test_preditc(model,
                                                          checkpoint_path,
                                                          )
        test_score=self.melg.get_reslut(epoch,probs,pred_labels,true_labels,csv_path)



