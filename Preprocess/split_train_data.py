# The Code is used to create training data subset
# not used in the project, use split_data_bucket instead

from sklearn.model_selection import train_test_split
import argparse
import os


def train_split(opt, data):
    used, unused = train_test_split(data, test_size=1-opt.subset_size, random_state=opt.seed)
    return used, unused


def write_data_txt(output_file, data):
    with open(output_file, 'w') as output:
        for line in data:
            question = line.strip().split('\t')[0]
            label = line.strip().split('\t')[1]
            output.write('{}\t{}\n'.format(question, label))


if __name__ == '__main__':
    main_arg_parser = argparse.ArgumentParser(description="parser")
    main_arg_parser.add_argument("-source_dir", type=str, default='../Data/GEO',
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
    test_source_path = os.path.join(args.source_dir, 'test.txt')  # only copy test.txt for now, expanded to dev later

    # TODO: add copy dev.txt to generalize (less priority)
    with open(source_path) as f:
        train_set = f.read().strip().split('\n')
    with open(test_source_path) as f:
        test_set = f.read().strip().split('\n')

    used_train_set, unused_train_set = train_split(args, train_set)
    print('number of subset of the training set used is {}'.format(len(used_train_set)))
    target_path = os.path.join(args.target_dir, 'train.txt')
    test_target_path = os.path.join(args.target_dir, 'test.txt')

    # Write data to the new folder
    write_data_txt(target_path, used_train_set)
    write_data_txt(test_target_path, test_set)




