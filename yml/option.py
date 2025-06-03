import os
import sys
sys.path.append('..')
from utils import get_logger
import yaml

logger = get_logger(__name__)
def parse(opt_path, is_train=True):
    '''
       opt_path: the path of yml file
       is_train: False
    '''
    logger.info('Reading .yml file .......')
    with open(opt_path,mode='r') as f:
        opt = yaml.load(f,Loader=yaml.FullLoader)

    # is_train into option
    opt['is_train'] = is_train

    return opt


if __name__ == "__main__":
    parse('./yml/dataset.yml')
