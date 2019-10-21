from Class.Lang import Lang
import argparse
import os


def init_lang(lang, data, question_type, phrase_list):
    """question_type: int 0 for question, int 1 for form"""
    for line in data:
        lang.add_sentence(line.strip().split('\t')[question_type], phrase_list)


def write_vocab_file(output_file, lang):
    keys = list(lang.w2i.keys())
    with open(output_file, 'w') as output:
        for key in keys:
            output.write('{}\t{}\n'.format(key, lang.w2c[key]))


main_arg_parser = argparse.ArgumentParser(description="parser")
main_arg_parser.add_argument("-data_dir", type=str, default="../Data/ATIS",
                             help="data dir, default set to GEO")  # Quick and Dirty Method to split GEO

args = main_arg_parser.parse_args()
data_path = os.path.join(args.data_dir, 'train.txt')
with open(data_path) as f:
    train_data = f.read().strip().split('\n')

question_lang = Lang()
init_lang(question_lang, train_data, 0, [])
print('number of question vocabulary is {}'.format(question_lang.n_words))
form_lang = Lang()
init_lang(form_lang, train_data, 1, [])
print('number of form vocabulary is {}'.format(form_lang.n_words))

out_q = os.path.join(args.data_dir, 'vocab.q.txt')
out_f = os.path.join(args.data_dir, 'vocab.f.txt')
write_vocab_file(out_q, question_lang)
write_vocab_file(out_f, form_lang)

