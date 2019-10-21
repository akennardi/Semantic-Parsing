from Src.seq2seq_attn import *
from Evaluation import Util
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# ---------- Plot Setup ---------- #
matplotlib.use('Agg')


def plot_acc(acc_transfer, acc_target, subsets, filename, title='Src Accuracy'):
    """
    Generate a plot from a model trained with all subsets and full training sample.
    The function will generate external plot file in ".pdf" format.
    :param acc_transfer: list of transfer learning accuracies
    :param acc_target: list of model without transfer learning accuracies
    :param subsets: list of subset
    :param filename: filename to save file
    :param title: title of plot
    :return: None
    """
    subsets = np.append(subsets, [100])
    plt.title(title, fontsize=15)
    plt.xlabel('Subset data fraction (%)', fontsize=15)
    plt.ylabel('Accuracy (%)', fontsize=15)
    plt.ylim(ymin=0, ymax=100)
    plt.grid(which='major')
    # loc, labels = plt.xticks(subsets, subsets)
    plt.plot(subsets, [i*100 for i in acc_transfer], 'r-D', markersize=6, label='transfer learning model',
             linewidth=2)
    plt.plot(subsets, [i*100 for i in acc_target], 'b-.^', markersize=6, label='target model',
             linewidth=2)
    for i, subset in enumerate(subsets):
        plt.text(subset, [i*100 for i in acc_transfer][i]+3, "{:.2f}".format(acc_transfer[i]*100), color='red')
        plt.text(subset, [i*100 for i in acc_target][i]-3, "{:.2f}".format(acc_target[i]*100), color='blue')
    plt.legend(loc='upper left')
    # plt.show()
    plt.savefig(filename+".pdf", format='pdf')
    plt.clf()


def plot_all_acc(opt, all_transfers, all_targets):
    """
    Generate plot from models trained with all subsets and full training sample in three different transfer learning
    setup.
    Function to generate plot in the report. The function will generate external plot file in ".pdf" format
    :param opt: argument parser args
    :param all_transfers: list of all model checkpoints from transfer learning
    :param all_targets: list of all model checkpoints without transfer learning
    :return: None
    """
    plt.figure(figsize=(15, 5))  # To be adjusted --> Size for ALTA (15,4)

    # Plot ATIS to Geo
    print("1st plot")
    plt.subplot(1, 3, 1)
    plt.grid(which='major')
    # plt.title("(a) ATIS to GeoQuery\n(Anonymised)", fontsize=15)
    plt.xlabel("Subset data fraction (%)", fontsize=15)
    plt.ylabel("Accuracy (%)", fontsize=15)
    plt.xlim(xmin=0, xmax=110)
    plt.ylim(ymin=-10, ymax=100)
    geo_dir = '../Data/GEO'
    all_geo_transfer, all_geo_target, geo_subset = get_list_acc(opt, all_transfers[0], all_targets[0],
                                                                opt.model_name, geo_dir)
    geo_subset = np.append(geo_subset, [100])
    plt.plot(geo_subset, [i * 100 for i in all_geo_transfer], 'r-D', markersize=6,
             linewidth=2)
    plt.plot(geo_subset, [i * 100 for i in all_geo_target], 'b-.^', markersize=6,
             linewidth=2)
    for i, subset in enumerate(geo_subset):
        plt.text(subset-4, [i * 100 for i in all_geo_transfer][i] + 8, "{:.2f}".format(all_geo_transfer[i] * 100),
                 color='red', fontsize=9)
        plt.text(subset-4, [i * 100 for i in all_geo_target][i] - 8, "{:.2f}".format(all_geo_target[i] * 100),
                 color='blue', fontsize=9)

    # Plot ATIS Un-anonymised to Geo Un-anonymised
    print("2nd plot")
    plt.subplot(1, 3, 2)
    plt.grid(which='major')
    # plt.title("(b) ATIS to GeoQuery\n(Un-anonymised)", fontsize=15)
    plt.xlabel("Subset data fraction (%)", fontsize=15)
    plt.xlim(xmin=0, xmax=110)
    plt.ylim(ymin=-10, ymax=100)
    geo_exp_dir = '../Data/GEO_EXP'
    all_geo_exp_transfer, all_geo_exp_target, geo_exp_subset = get_list_acc(opt, all_transfers[1], all_targets[1],
                                                                            opt.model_name, geo_exp_dir)
    geo_exp_subset = np.append(geo_exp_subset, [100])
    plt.plot(geo_exp_subset, [i * 100 for i in all_geo_exp_transfer], 'r-D', markersize=6,
             linewidth=2)
    plt.plot(geo_exp_subset, [i * 100 for i in all_geo_exp_target], 'b-.^', markersize=6,
             linewidth=2)
    for i, subset in enumerate(geo_exp_subset):
        plt.text(subset-4, [i * 100 for i in all_geo_exp_transfer][i] + 8, "{:.2f}".format(all_geo_exp_transfer[i] * 100),
                 color='red', fontsize=9)
        plt.text(subset-4, [i * 100 for i in all_geo_exp_target][i] - 8, "{:.2f}".format(all_geo_exp_target[i] * 100),
                 color='blue', fontsize=9)

    print("3rd plot")
    plt.subplot(1, 3, 3)
    plt.grid(which='major')
    # plt.title("(c) ATIS to GeoQuery\n(Un-anonymised with Query-Split)", fontsize=15)
    plt.xlabel("Subset data fraction (%)", fontsize=15)
    plt.xlim(xmin=0, xmax=110)
    plt.ylim(ymin=-10, ymax=100)
    geo_exp_query_dir = geo_dir = '../Data/GEO_EXP_QUERY'
    all_geo_exp_query_transfer, all_geo_exp_query_target, geo_exp_query_subset = \
        get_list_acc(opt, all_transfers[2], all_targets[2], opt.model_name, geo_exp_query_dir)
    geo_exp_query_subset = np.append(geo_exp_query_subset, [100])
    plt.plot(geo_exp_query_subset, [i * 100 for i in all_geo_exp_query_transfer], 'r-D', markersize=6,
             linewidth=2, label='transfer learning model')
    plt.plot(geo_exp_query_subset, [i * 100 for i in all_geo_exp_query_target], 'b-.^', markersize=6,
             linewidth=2, label='target model\nwithout transfer')
    for i, subset in enumerate(geo_exp_query_subset):
        plt.text(subset-4, [i * 100 for i in all_geo_exp_query_transfer][i] + 8,
                 "{:.2f}".format(all_geo_exp_query_transfer[i] * 100),
                 color='red', fontsize=9)
        plt.text(subset-4, [i * 100 for i in all_geo_exp_query_target][i] - 8,
                 "{:.2f}".format(all_geo_exp_query_target[i] * 100),
                 color='blue', fontsize=9)
    plt.legend(loc='upper right')

    plt.tight_layout()
    filename = './Plot/Transfer_Learning_All'
    plt.savefig(filename+".pdf", format='pdf')
    plt.clf()


def convert_to_string(idx_list, form_manager):
    """
    Function to convert the id into string of meaning representation.
    Reference: https://github.com/Alex-Fabbri/lang2logic-PyTorch
    :param idx_list: list of id sequence
    :param form_manager: meaning representation manager from symbol manager class
    :return: Meaning representation in String
    """
    w_list = []
    for i in range(len(idx_list)):
        w_list.append(form_manager.get_idx_symbol(int(idx_list[i])))
    return " ".join(w_list)


def do_generate(encoder, decoder, attention_decoder, enc_w_list, word_manager, form_manager, opt, using_gpu, checkpoint):
    """
    Function to generate the meaning representation from the model stored in the checkpoint
    Reference: https://github.com/Alex-Fabbri/lang2logic-PyTorch
    :param encoder: encoder LSTM model
    :param decoder: decoder LSTM model
    :param attention_decoder: attention decoder model
    :param enc_w_list: list of input sequence
    :param word_manager: input sequence manager from symbol manager class
    :param form_manager: meaning representation manager from symbol manager class
    :param opt: argument parser
    :param using_gpu: argument indicating GPU usage
    :param checkpoint: checkpoint folder where the model is stored
    :return: prediction of meaning representation
    """
    # initialize the rnn state to all zeros
    enc_w_list.append(word_manager.get_symbol_idx('<S>'))
    enc_w_list.insert(0, word_manager.get_symbol_idx('<E>'))
    end = len(enc_w_list)
    prev_c = torch.zeros((1, encoder.hidden_size), requires_grad=False)
    prev_h = torch.zeros((1, encoder.hidden_size), requires_grad=False)
    enc_outputs = torch.zeros((1, end, encoder.hidden_size), requires_grad=False)
    if using_gpu:
        prev_c = prev_c.cuda()
        prev_h = prev_h.cuda()
        enc_outputs = enc_outputs.cuda()
    # reversed order
    for i in range(end-1, -1, -1):
        cur_input = torch.tensor(np.array(enc_w_list[i]), dtype=torch.long)
        if using_gpu:
            cur_input = cur_input.cuda()
        prev_c, prev_h = encoder(cur_input, prev_c, prev_h)
        enc_outputs[:, i, :] = prev_h
    # encoder_outputs = torch.stack(encoder_outputs).view(-1, end, encoder.hidden_size)
    # decode
    if opt.sample == 0 or opt.sample == 1:
        text_gen = []
        if opt.gpuid >= 0:
            prev_word = torch.tensor([form_manager.get_symbol_idx('<S>')], dtype=torch.long).cuda()
        else:
            prev_word = torch.tensor([form_manager.get_symbol_idx('<S>')], dtype=torch.long)
        while True:
            prev_c, prev_h = decoder(prev_word, prev_c, prev_h)
            pred = attention_decoder(enc_outputs, prev_h)
            # print("prediction: {}\n".format(pred))
            # log probabilities from the previous timestamp
            if opt.sample == 0:
                # use argmax
                _, _prev_word = pred.max(1)
                prev_word = _prev_word.reshape(1)
            if (prev_word[0] == form_manager.get_symbol_idx('<E>')) or (len(text_gen) >= checkpoint["opt"].dec_seq_length):
                break
            else:
                text_gen.append(prev_word[0])
        return text_gen


def get_all_acc(all_models, opt, subsets, data_dir):
    """
    Function to copmpute accuracies of all model trained with subset data set
    :param all_models: list of all models (checkpoint folder)
    :param opt: argument parser
    :param subsets: list of subset {10,20,...,90}
    :param data_dir: directory of the data used
    :return: list of model accuracy of every subset
    """
    all_acc = []
    for n, model in enumerate(all_models):
        checkpoint = model
        encoder = checkpoint["encoder"]
        decoder = checkpoint["decoder"]
        attention_decoder = checkpoint["attention_decoder"]
        # put in eval mode for dropout
        encoder.eval()
        decoder.eval()
        attention_decoder.eval()
        # initialize the vocabulary manager to display text
        managers = pkl.load(open("{}/map.pkl".format(data_dir + '_' + str(subsets[n])), "rb"))
        word_manager, form_manager = managers
        # load data
        data = pkl.load(open("{}/test.pkl".format(data_dir + '_' + str(subsets[n])), "rb"))
        reference_list = []
        candidate_list = []

        for i in range(len(data)):
            # print(i)
            x = data[i]
            reference = x[1]
            candidate = do_generate(encoder, decoder, attention_decoder, x[0], word_manager, form_manager, opt,
                                    using_gpu, checkpoint)
            candidate = [int(c) for c in candidate]

            num_left_paren = sum(1 for c in candidate if form_manager.idx2symbol[int(c)] == "(")
            num_right_paren = sum(1 for c in candidate if form_manager.idx2symbol[int(c)] == ")")
            diff = num_left_paren - num_right_paren
            # print(diff)
            if diff > 0:
                for j in range(diff):
                    candidate.append(form_manager.symbol2idx[")"])
            elif diff < 0:
                candidate = candidate[:diff]

            ref_str = convert_to_string(reference, form_manager)
            cand_str = convert_to_string(candidate, form_manager)

            reference_list.append(reference)
            candidate_list.append(candidate)
            # print to console
            if opt.display > 0:
                print("results: ")
                print('reference:{}'.format(ref_str))
                print('predicted:{}'.format(cand_str))
                print(' ')

        val_acc = Util.compute_tree_accuracy(candidate_list, reference_list, form_manager)
        all_acc.append(val_acc)
    return all_acc


def get_acc(model, opt, data_dir):
    """
    Compute accuracy of a model
    :param model: model from checkpoint
    :param opt:  argument parser
    :param data_dir: directory of the data used for experiment
    :return: accuracy
    """
    checkpoint = model
    encoder = checkpoint["encoder"]
    decoder = checkpoint["decoder"]
    attention_decoder = checkpoint["attention_decoder"]
    # put in eval mode for dropout
    encoder.eval()
    decoder.eval()
    attention_decoder.eval()
    managers = pkl.load(open("{}/map.pkl".format(data_dir), "rb"))
    word_manager, form_manager = managers
    # load data
    data = pkl.load(open("{}/test.pkl".format(data_dir), "rb"))
    reference_list = []
    candidate_list = []

    for i in range(len(data)):
        # print(i)
        x = data[i]
        reference = x[1]
        candidate = do_generate(encoder, decoder, attention_decoder, x[0], word_manager, form_manager, opt,
                                using_gpu, checkpoint)
        candidate = [int(c) for c in candidate]

        num_left_paren = sum(1 for c in candidate if form_manager.idx2symbol[int(c)] == "(")
        num_right_paren = sum(1 for c in candidate if form_manager.idx2symbol[int(c)] == ")")
        diff = num_left_paren - num_right_paren
        # print(diff)
        if diff > 0:
            for j in range(diff):
                candidate.append(form_manager.symbol2idx[")"])
        elif diff < 0:
            candidate = candidate[:diff]

        ref_str = convert_to_string(reference, form_manager)
        cand_str = convert_to_string(candidate, form_manager)

        reference_list.append(reference)
        candidate_list.append(candidate)
        # print to console
        if opt.display > 0:
            print("results: ")
            print('reference:{}'.format(ref_str))
            print('predicted:{}'.format(cand_str))
            print(' ')

        val_acc = Util.compute_tree_accuracy(candidate_list, reference_list, form_manager)
    return val_acc


def get_list_acc(opt, model_transfer, model_target, model_name, data_dir):
    """
    Compute accuracies of all model.
    :param opt: argument parser
    :param model_transfer: checkpoint of source task model
    :param model_target: checkpoint of target task model
    :param model_name: name of the model
    :param data_dir: directory of the data used for experiment
    :return: list of accuracies as 3 lists.
    """
    start = 100 / args.subset_split
    all_data_subset = np.linspace(start, 100 - start, (opt.subset_split - 1),
                                  dtype=int)  # [10 20 30 40 50 60 70 80 90]
    transfer_models = []
    target_models = []
    transfer_path = model_transfer
    target_path = model_target
    test_model_name = model_name

    for size in all_data_subset:
        path_transfer = transfer_path.replace('xx', str(size)) + test_model_name
        transfer_models.append(torch.load(path_transfer))
        path_target = target_path.replace('xx', str(size)) + test_model_name
        target_models.append(torch.load(path_target))

    all_acc_transfer = get_all_acc(transfer_models, opt, all_data_subset, data_dir)
    all_acc_target = get_all_acc(target_models, opt, all_data_subset, data_dir)

    path_transfer_all = transfer_path.replace('_xx', '') + test_model_name
    full_transfer = torch.load(path_transfer_all)
    full_transfer_acc = get_acc(full_transfer, opt, data_dir)
    transfer_models.append(full_transfer)
    all_acc_transfer.append(full_transfer_acc)

    path_target_all = target_path.replace('_xx', '') + test_model_name
    full_target = torch.load(path_target_all)
    full_target_acc = get_acc(full_target, opt, data_dir)
    target_models.append(full_target)
    all_acc_target.append(full_target_acc)
    return all_acc_transfer, all_acc_target, all_data_subset


main_arg_parser = argparse.ArgumentParser(description="parser")
main_arg_parser.add_argument('-gpuid', type=int, default=-1, help='which gpu to use. -1 = use CPU')
main_arg_parser.add_argument('-data_dir', type=str, default='../Data/GEO_EXP', help='data path')
main_arg_parser.add_argument('-sample', type=int, default=0, help='0 to use max at each time step')
main_arg_parser.add_argument('-display', type=int, default=0, help='whether display on console')
main_arg_parser.add_argument('-model_transfer', type=str, default='../Src/checkpoint_atis_geo_xx_transfer')
main_arg_parser.add_argument('-model_target', type=str, default='../Src/checkpoint_geo_xx')
main_arg_parser.add_argument('-model_transfer2', type=str, default='../Src/checkpoint_atis_geo_exp_xx_transfer')
main_arg_parser.add_argument('-model_target2', type=str, default='../Src/checkpoint_geo_exp_xx')
main_arg_parser.add_argument('-model_transfer3', type=str, default='../Src/checkpoint_atis_geo_exp_query_xx_transfer')
main_arg_parser.add_argument('-model_target3', type=str, default='../Src/checkpoint_geo_exp_query_xx')
main_arg_parser.add_argument('-plot_all', type=int, default=0,
                             help='0 plot 1 figure, 1 plot everything as sub-figure')
main_arg_parser.add_argument('-model_name', type=str, default='/model_seq2seq_attention')
main_arg_parser.add_argument('-subset_split', type=int, default=10)
main_arg_parser.add_argument('-filename', type=str, default='./Plot/Transfer_Learning_ATIS_GEO_EXP')
main_arg_parser.add_argument('-title', type=str, default='Src Accuracy')

args = main_arg_parser.parse_args()
using_gpu = False
if args.gpuid > -1:
    using_gpu = True
# load all checkpoint model
plot_sub_figure = False
if args.plot_all == 1:
    plot_sub_figure = True

if not plot_sub_figure:
    print('Now plotting...')
    acc_transfers, acc_targets, data_subsets = get_list_acc(args, args.model_transfer,
                                                            args.model_target, args.model_name, args.data_dir)
    plot_acc(acc_transfers, acc_targets, data_subsets, args.filename, args.title)
    print("Graph is stored as: {}".format(args.filename))
    print('Done!')

if plot_sub_figure:
    print("Now plotting...")
    list_transfer_models = [args.model_transfer, args.model_transfer2, args.model_transfer3]
    list_target_models = [args.model_target, args.model_target2, args.model_target3]
    plot_all_acc(args, list_transfer_models, list_target_models)
    print('Done!')




