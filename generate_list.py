import numpy as np
import os
import soundfile as sf
from random import shuffle
from create_args import *


def generate_list(args):

    # read the whole timit database and create a complete list of speech files
    task_path = os.path.join(args.TIMIT_path, args.task)

    # initialize lists
    files_list = []
    good_files_list = []

    # create files list

    subdirs = [d for d in os.listdir(os.path.join(args.TIMIT_path, args.task)) if os.path.isdir(os.path.join(os.path.join(args.TIMIT_path, args.task), d))]
    
    for subdr in subdirs:

        for subsubdir in os.listdir(os.path.join(args.TIMIT_path, args.task, subdr)):
        
            files = os.listdir(os.path.join(args.TIMIT_path, args.task, subdr, subsubdir))

            for file in files:
                
                if file.endswith(".wav"):
                    print(file)
                    
                    files_list.append(os.path.join(args.TIMIT_path, args.task, subdr, subsubdir, file))

    count = 0
    # delete invalid files
    for path in files_list:
    
        try:
            data = sf.read(path)
            good_files_list.append(path)
            del data
        
        except Exception:
            continue

        count +=1
        print(count)

    # shuffle good_files_list
    if args.task == 'train':
        shuffle(good_files_list)

    # save good_files_list to .txt file
    with open(os.path.join(txt_path, args.task + '.txt'), 'w') as f:
        for line in good_files_list:
            f.write(f"{line}\n")


if __name__ == '__main__':

    args, config = create_args()
    
    # create txt path
    txt_path = os.path.join(args.WORK_PATH, 'txt')
    if not os.path.exists(txt_path):
        os.makedirs(txt_path)
    
    generate_list(args)

