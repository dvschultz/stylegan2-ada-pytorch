# No QA
# Author: JulianPinzaru

import argparse
import os
import random
import json


def parse_args():
    desc = "Tool to create multiclass json labels file for stylegan2-ada-pytorch"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--verbose', action='store_true',
        help='Print progress to console.')

    parser.add_argument('--input_folder', type=str,
        default='./input/',
        help='Directory path to the inputs folder. (default: %(default)s)')

    parser.add_argument('--output_folder', type=str,
        default='./output/',
        help='Directory path to the outputs folder. (default: %(default)s)')

    args = parser.parse_args()
    return args


def main():
    global args
    args = parse_args()
    base_dir = ''

    remakePath = args.output_folder
    if not os.path.exists(remakePath):
        os.makedirs(remakePath)

    data_dict = {}
    data_dict['labels'] = []
    label_counter = 0

    with open(os.path.join(remakePath, 'dataset.json'), 'w') as outfile:

        for root, subdirs, files in os.walk(args.input_folder):
            subdirs.sort(key=int)    
            if len(subdirs) > 0:
                base_dir = root
                continue

            current_subdir = os.path.split(root)[1]

            for filename in files:
                file_path = os.path.join(current_subdir, filename)
                if(args.verbose): print('\t- file %s (full path: %s)' % (filename, file_path))
                data_dict['labels'].append([file_path, label_counter])

            label_counter += 1

        json.dump(data_dict, outfile)


if __name__ == "__main__":
    main()

