import argparse
from Config.base import *




def get_args(dataname):

    argp = argparse.ArgumentParser()
    argp.add_argument("--dataname", default=dataname,type=str)
    argp.add_argument("--model_name",default="STAS_MIL")
    argp.add_argument("--train_data", default=train_data_path[dataname], type=str)
    argp.add_argument("--test_data", default=test_data_path[dataname], type=str)
    argp.add_argument("--nclass", default=2, type=int)

    argp.add_argument("--input_dim", default=768, type=int)
    argp.add_argument("--embed_dim", default=256, type=int)
    argp.add_argument("--attn_dropout",default=0.0,type=float)
    argp.add_argument("--attn_emb",default=64,type=float)

    argp.add_argument("--th", default=0.5, type=float)
    argp.add_argument("--scale",default=0.5,type=float)
    argp.add_argument("--add_margin",default=True,type=float)

    argp.add_argument("--train_bs", default=12)
    argp.add_argument("--test_bs", default=1)

    argp.add_argument("--num_workers", default=4)
    argp.add_argument("--lr", default=2e-4)
    argp.add_argument("--wd", default=1e-5)
    argp.add_argument("--start_epoch", default=0, type=int)
    argp.add_argument("--num_epochs", default=100, type=int)
    argp.add_argument("--patient", default=50)

    argp.add_argument("--start_seed",default=0,type=int)
    argp.add_argument("--end_seed",default=5,type=int)
    argp.add_argument("--epoch_frq",default=-1,type=str)

    argp.add_argument("--tensorboard_dir", default="./recoder/{}/log".format(dataname))
    argp.add_argument("--checkpoint_dir", default="./recoder/{}/checkpoint".format(dataname))
    argp.add_argument("--metic_dir", default="./recoder/{}/results".format(dataname))
    argp.add_argument("--metric",default="acc")
    argp.add_argument("--seed", default=0)
    argp.add_argument("--device", default="cuda")
    args = argp.parse_args()
    return args