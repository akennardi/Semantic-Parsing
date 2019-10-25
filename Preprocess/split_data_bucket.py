from Class.Lang import Lang
import argparse
import os
from sklearn.utils import shuffle

"""
Script to separate data set (GeoQuery) into 10 subset
"""


def get_subset(data, opt):
    last = int(opt.subset_size * (len(train_set))) + 1
    print(last)
    subset = data[:last]
    print(len(subset))
    return subset


def write_data_txt(output_file, data):
    with open(output_file, 'w') as output:
        for line in data:
            question = line.strip().split('\t')[0]
            label = line.strip().split('\t')[1]
            output.write('{}\t{}\n'.format(question, label))


if __name__ == '__main__':
    main_arg_parser = argparse.ArgumentParser(description="parser")
    main_arg_parser.add_argument("-source_dir", type=str, default='../Data/GEO_EXP',
                                 help='source directory')
    main_arg_parser.add_argument("-target_dir", type=str, default='../Data/GEO_EXP_10',
                                 help='target directory')
    main_arg_parser.add_argument("-subset_size", type=float, default=0.1)
    main_arg_parser.add_argument("-seed", type=int, default=123,
                                 help='random_seed')

    args = main_arg_parser.parse_args()
    # create target dir if the directory is not exist
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)

    source_path = os.path.join(args.source_dir, 'train.txt')
    test_source_path = os.path.join(args.source_dir, 'test.txt')

    with open(source_path) as f:
        train_set = f.read().strip().split('\n')
    with open(test_source_path) as f:
        test_set = f.read().strip().split('\n')

    print("number of training set:{}".format(len(train_set)))
    train_set = shuffle(train_set, random_state=123)
    subset_train_set = get_subset(train_set, args)
    print('number of subset of the training set used is {}'.format(len(subset_train_set)))

    target_path = os.path.join(args.target_dir, 'train.txt')
    test_target_path = os.path.join(args.target_dir, 'test.txt')

    write_data_txt(target_path, subset_train_set)
    write_data_txt(test_target_path, test_set)




