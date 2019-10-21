from sklearn.model_selection import train_test_split
from Class.Lang import Lang
import argparse
import os

test_path = '../Data/ATIS_EXP/atis-logic-exp.txt'

# TODO: change back all separator to ' ||| '


def data_split(path, val_size, test_size):
    with open(path) as f:
        data = f.read().strip().split('\n')

    num_test_set = test_size/(test_size + val_size)
    train, test_val = train_test_split(data, test_size=val_size+test_size, random_state=1)
    test, val = train_test_split(test_val, test_size=num_test_set, random_state=1)
    return train, test, val


def init_lang(lang, data, question_type, phrase_list):
    """question_type: int 0 for question, int 1 for form"""
    for line in data:
        lang.add_sentence(line.strip().split(' ||| ')[question_type], phrase_list)


def write_vocab_file(output_file, lang):
    keys = list(lang.w2i.keys())
    with open(output_file, 'w') as output:
        for key in keys:
            output.write('{}\t{}\n'.format(key, lang.w2c[key]))


def write_data_txt(output_file, data):
    with open(output_file, 'w') as output:
        for line in data:
            question = line.strip().split(' ||| ')[0]
            label = line.strip().split('||| ')[1]
            output.write('{}\t{}\n'.format(question, label))


main_arg_parser = argparse.ArgumentParser(description="parser")
main_arg_parser.add_argument("-data_dir", type=str, default="../Data/ATIS_EXP",
                             help="data dir, default set to ATIS_EXP")  # Quick and Dirty Method to split GEO
main_arg_parser.add_argument("-test_size", type=float, default=0.1,
                             help="test set size")
main_arg_parser.add_argument("-val_size", type=float, default=0.1,
                             help='validation set size')  # default 0.1


args = main_arg_parser.parse_args()
data_path = os.path.join(args.data_dir, 'atis-logic-exp.txt')
train_data, test_data, dev_data = data_split(data_path, args.val_size, args.test_size)
place_list = [('san', 'francisco'), ('new', 'york'), ('san', 'diego'), ('st.', 'petersburg'), ('san', 'jose'),
              ('new', 'jersey'), ('salt', 'lake', 'city'), ('los', 'angeles')]
question_lang = Lang()
init_lang(question_lang, train_data, 0, place_list)
print('number of question vocabulary is {}'.format(question_lang.n_words))
form_lang = Lang()
init_lang(form_lang, train_data, 1, place_list)
print('number of form vocabulary is {}'.format(form_lang.n_words))

# write vocabulary file
out_q = os.path.join(args.data_dir, 'vocab.q.txt')
out_f = os.path.join(args.data_dir, 'vocab.f.txt')
write_vocab_file(out_q, question_lang)
write_vocab_file(out_f, form_lang)

# write train, dev, test files
out_train = os.path.join(args.data_dir, 'train.txt')
out_test = os.path.join(args.data_dir, 'test.txt')
out_dev = os.path.join(args.data_dir, 'dev.txt')
# write_data_txt(out_train, train_data)
# write_data_txt(out_test, test_data)
# write_data_txt(out_dev, dev_data)




