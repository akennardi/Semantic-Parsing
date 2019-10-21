import os
import argparse
import json


def write_txt_file(output_file, data_list):
    with open(output_file, 'w') as output:
        for d in data_list:
            output.write("{}\n".format(d))


main_arg_parser = argparse.ArgumentParser(description="parser")
main_arg_parser.add_argument("-data_dir", type=str, default="../Data/GEO_EXP",
                             help="data dir, default set to GEO EXP folder")
main_arg_parser.add_argument("-target_dir", type=str, default="../Data/GEO_EXP_QUERY",
                             help='target dir to save query split data')
# TODO: add option to create new folder for query split (under GEO _EXP folder, it is based on question split)

args = main_arg_parser.parse_args()
geo_path = os.path.join(args.data_dir, 'geography-logic.json')

# open GEO data set
with open(geo_path) as f:
    data = json.load(f)

# parse GEO json data set
train_data = []
dev_data = []
test_data = []
query_train_data = []
query_dev_data = []
query_test_data = []
all_data = []
for i in range(len(data)):
    # print(data[i])
    for j in range(len(data[i]['sentences'])):
        question = data[i]['sentences'][j]['text']
        # var0 = data[i]['sentences'][j]['variables']['var0']
        logic = data[i]['logic'][0]
        if (len(data[i]['sentences'][j]['variables'])) == 0:
            exp_question = question
            exp_logic = logic

            # create whole data-set
            all_data.append("{}\t{}".format(exp_question, exp_logic))

            # create query-split (Finegan-Dollak et all, 2018)
            if data[i]['query-split'] == 'train':
                query_train_data.append("{}\t{}".format(exp_question, exp_logic))
            elif data[i]['query-split'] == 'dev':
                query_dev_data.append("{}\t{}".format(exp_question, exp_logic))
            else:
                query_test_data.append("{}\t{}".format(exp_question, exp_logic))

            # create question-split(Zettlemoyer and Collins, 2005, 2007)
            if data[i]['sentences'][j]['question-split'] == 'train':
                train_data.append("{}\t{}".format(exp_question, exp_logic))
            elif data[i]['sentences'][j]['question-split'] == 'dev':
                dev_data.append("{}\t{}".format(exp_question, exp_logic))
            else:
                test_data.append("{}\t{}".format(exp_question, exp_logic))
        else:
            exp_question = question
            exp_logic = logic
            for k, var in enumerate(data[i]['sentences'][j]['variables']):  # to handle question with more than one var
                v = data[i]['sentences'][j]['variables'][var]
                exp_question = (exp_question.replace('var{}'.format(k), v))
                exp_logic = (exp_logic.replace('var{}'.format(k), v))

            # create whole data-set
            all_data.append("{}\t{}".format(exp_question, exp_logic))

            # create query-split (Finegan-Dollak et all, 2018)
            if data[i]['query-split'] == 'train':
                query_train_data.append("{}\t{}".format(exp_question, exp_logic))
            elif data[i]['query-split'] == 'dev':
                query_dev_data.append("{}\t{}".format(exp_question, exp_logic))
            else:
                query_test_data.append("{}\t{}".format(exp_question, exp_logic))

            # create question-split(Zettlemoyer and Collins, 2005, 2007)
            if data[i]['sentences'][j]['question-split'] == 'train' or data[i]['sentences'][j]['question-split'] == 'dev':
                train_data.append("{}\t{}".format(exp_question, exp_logic))
            elif data[i]['sentences'][j]['question-split'] == 'dev':
                dev_data.append("{}\t{}".format(exp_question, exp_logic))
            else:
                test_data.append("{}\t{}".format(exp_question, exp_logic))


# write question split data under folder GEO_EXP
all_data_file = os.path.join(args.data_dir, 'geography-logic-exp.txt')
train_data_file = os.path.join(args.data_dir, 'train.txt')
dev_data_file = os.path.join(args.data_dir, 'dev.txt')
test_data_file = os.path.join(args.data_dir, 'test.txt')

write_txt_file(all_data_file, all_data)
write_txt_file(train_data_file, train_data)
write_txt_file(dev_data_file, dev_data)
write_txt_file(test_data_file, test_data)

# write query split data under folder GEO_EXP_QUERY
if not os.path.exists(args.target_dir):
    os.makedirs(args.target_dir)

query_train_file = os.path.join(args.target_dir, 'train.txt')
query_dev_file = os.path.join(args.target_dir, 'dev.txt')
query_test_file = os.path.join(args.target_dir, 'test.txt')

write_txt_file(query_train_file, query_train_data)
write_txt_file(query_dev_file, query_dev_data)
write_txt_file(query_test_file, query_test_data)

print("numbers of unique query:{}".format(len(data)))
print("all data points: {}".format(len(all_data)))
print("number of training data in query split: {}".format(len(query_train_data)))
print("number of dev data in query split: {}".format(len(dev_data)))
print("number of test data in query split: {}".format(len(query_test_data)))
print("number of training data in question split: {}".format(len(train_data)))
print("number of dev data in question split: {}".format(len(dev_data)))
print("number of test data in question split: {}".format(len(test_data)))












