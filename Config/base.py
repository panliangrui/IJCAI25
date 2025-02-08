import os
from club import utils


class_dict={
    "CPTAC":2,
    "TCGA":2,
    "CSU":2,
}

train_data_path = {data_name:os.path.join("/root/autodl-tmp/Datasets/",data_name,"pt_files") for data_name in class_dict.keys()}
test_data_path = {data_name:os.path.join("/root/autodl-tmp/Datasets/",data_name,"pt_files") for data_name in class_dict.keys()}

def set_data(args):
    utils.fix_random_seed(args.seed)
    csv_seed, csv_fold = args.seed // 5, args.seed % 5

    excel_path =  {data_name:os.path.join("/root/autodl-tmp/IJAI/data/",
                                          data_name,
                                          f"file_labels_seed{csv_seed}.xlsx") for data_name in class_dict.keys()}

    args.excel_path = excel_path[args.dataname]
    args.train_sheet = "train_fold{}".format(csv_fold)
    args.test_sheet = "test_fold{}".format(csv_fold)

    args.test_csv_path = "{}/test_on_kflod_{}.csv".format(args.metic_dir, args.dataname)

    args.checkpoint_path = os.path.join(args.checkpoint_dir,
                                   "{}_best_seed{}.pth".format(args.model_name, args.seed))
def reset_run_root(run_folder,args):
    if not os.path.exists(run_folder):
        os.makedirs(run_folder)
    args.tensorboard_dir=os.path.join(run_folder,args.dataname, "tensorboard")
    args.checkpoint_dir=os.path.join(run_folder,args.dataname, "checkpoint")
    args.metic_dir=os.path.join(run_folder,args.dataname, "metric")

    if not os.path.exists(args.tensorboard_dir):
        os.makedirs(args.tensorboard_dir)

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    if not os.path.exists(args.metic_dir):
        os.makedirs(args.metic_dir)
