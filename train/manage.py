from Config import args_base
from Config.base import *
from club import train_test


def train_th(dataname,th,scale):
    args = args_base.get_args(dataname=dataname)
    args.th = th
    args.scale = scale
    run_root = "./th_{}_factor_{}".format(args.th, args.scale)
    reset_run_root(run_root, args)

    train_test.main(args, train=True, test=True)
def main():
    train_th("CPTAC", 0.5, 0.3)
    train_th("TCGA", 0.5, 0.3)
    train_th("CSU", 0.5, 0.3,)


if __name__ == '__main__':
    main()