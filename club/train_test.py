from club.Lightning import Lightning
from model.STAS_MILP import STAS_MIL
from Config.base import *


def main(args,train,test):
    for i in range(args.start_seed,args.end_seed):
        args.seed=i
        set_data(args)
        model = STAS_MIL(scale=args.scale,th=args.th,add_margin=args.add_margin)

        lg = Lightning(args)
        if train:
            lg.train(model=model)

        if test:
            test_epoch = 0 if i == 0 else 200
            lg.test(epoch=test_epoch,
                    model=model,
                    checkpoint_path=args.checkpoint_path,
                    csv_path=args.test_csv_path)
