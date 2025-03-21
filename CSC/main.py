import torch
import argparse
from config.voc import get_args_parser
from pathlib import Path
import sys
import numpy as np
import random
from CSC import CSC_MLCIL
# from t_SNE import CSC_MLCIL

def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    CSC = CSC_MLCIL(args)
    CSC.train_test() 
           
if __name__ == '__main__':
    parser = argparse.ArgumentParser('CSC training and evaluation configs')
    get_args_parser(parser)
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
    sys.exit(0)
