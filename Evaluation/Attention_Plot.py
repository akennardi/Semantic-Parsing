from Model.seq2seq_attn import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns


def convert_to_string(idx_list, form_manager):
    w_list = []
    for i in range(len(idx_list)):
        w_list.append(form_manager.get_idx_symbol(int(idx_list[i])))
    return " ".join(w_list)


def do_generate(encoder, decoder, attention_decoder, enc_w_list, word_manager, form_manager, opt, using_gpu, checkpoint):
    # initialize the rnn state to all zeros
    enc_w_list.append(word_manager.get_symbol_idx('<S>'))
    enc_w_list.insert(0, word_manager.get_symbol_idx('<E>'))
    end = len(enc_w_list)
    print('enc_w_list:')
    print(enc_w_list)
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
        dots = []
        while True:
            prev_c, prev_h = decoder(prev_word, prev_c, prev_h)
            pred = attention_decoder(enc_outputs, prev_h)
            dots.append(torch.bmm(enc_outputs, prev_h.unsqueeze(2)))
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
        return text_gen, dots


def get_alignment_score(checkpoint, data_dir, idx, opt, using_gpu=False):
    """idx: sample index"""
    # encoder, decoder, attention is inside
    encoder = checkpoint["encoder"]
    decoder = checkpoint["decoder"]
    attention_decoder = checkpoint["attention_decoder"]
    # put in eval mode for dropout
    encoder.eval()
    decoder.eval()
    attention_decoder.eval()
    # initialize the vocabulary manager to display text
    managers = pkl.load(open("{}/map.pkl".format(data_dir), "rb"))
    word_manager, form_manager = managers
    # load data
    data = pkl.load(open("{}/test.pkl".format(data_dir), "rb"))
    reference_list = []
    candidate_list = []
    x = data[idx]
    reference = x[1]  # index 0 for sentence, index 1 for reference of meaning representation
    candidate, score = do_generate(encoder, decoder, attention_decoder, x[0], word_manager, form_manager, opt,
                            using_gpu, checkpoint)
    print(len(candidate))
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

    print('reference:{}'.format(ref_str))
    print('predicted:{}'.format(cand_str))
    print(x[1])
    print(candidate_list[0])
    test_x = form_manager.get_idx_symbol(candidate_list[0][1])
    print(cand_str.split())
    print(test_x)
    question_str = []
    for i in range(len(x[0])):
        question_str.append(word_manager.get_idx_symbol(x[0][i]))
    print(question_str)
    return score, question_str, cand_str.split()


def plot_attention(checkpoint, data_dir, idx, opt, using_gpu=False):
    score, q, f = get_alignment_score(checkpoint, data_dir, idx, opt)
    out_len = len(f)
    in_len = score[0].squeeze().size()[0]
    score_tensor = torch.zeros(out_len, in_len)
    for i in range(out_len):
        score_tensor[i, :] = score[i].squeeze()
    score_tensor = score_tensor.detach().numpy()
    row_sums = score_tensor.sum(axis=1)
    score_tensor = score_tensor / row_sums[:, np.newaxis]

    fig = plt.figure(figsize=(10, 10))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(score_tensor, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5},
                xticklabels=q, yticklabels=f)

    plt.show()
    plt.clf()


test_path = '../Model/checkpoint_dir_atis/model_seq2seq_attention'
model = torch.load(test_path)

main_arg_parser = argparse.ArgumentParser(description="parser")
main_arg_parser.add_argument('-gpuid', type=int, default=-1, help='which gpu to use. -1 = use CPU')
main_arg_parser.add_argument('-data_dir', type=str, default='../Data/ATIS', help='data path')
main_arg_parser.add_argument('-sample', type=int, default=0, help='0 to use max at each time step')

args = main_arg_parser.parse_args()
data_dir = args.data_dir
using_gpu = False
if args.gpuid > -1:
    using_gpu = True

plot_attention(model, data_dir, 10, args, using_gpu)





