from Src.seq2seq_attn import *


def eval_training(opt, train_loader, encoder, decoder, attention_decoder, encoder_optimizer, decoder_optimizer,
                  attention_decoder_optimizer, criterion, using_gpu, form_manager):
    # encode, decode, backward, return loss
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    attention_decoder_optimizer.zero_grad()
    enc_batch, enc_len_batch, dec_batch = train_loader.random_batch()
    # do not predict after <E>
    enc_max_len = enc_batch.size(1)
    dec_max_len = dec_batch.size(1) - 1

    enc_outputs = torch.zeros((enc_batch.size(0), enc_max_len, encoder.hidden_size), requires_grad=False)
    if using_gpu:
        enc_outputs = enc_outputs.cuda()

    enc_s = {}
    for j in range(opt.enc_seq_length + 1):
        enc_s[j] = {}

    dec_s = {}
    for j in range(opt.dec_seq_length + 1):
        dec_s[j] = {}

    for i in range(1, 3):  # 1, 2 are index for cell and hidden, respectively. Initialization of hidden unit
        enc_s[0][i] = torch.zeros((opt.batch_size, opt.rnn_size), dtype=torch.float, requires_grad=False)
        dec_s[0][i] = torch.zeros((opt.batch_size, opt.rnn_size), dtype=torch.float, requires_grad=False)
        if using_gpu:
            enc_s[0][i] = enc_s[0][i].cuda()
            dec_s[0][i] = dec_s[0][i].cuda()

    for i in range(enc_max_len):  # unroll the encoder, iteration over training sequence
        enc_s[i+1][1], enc_s[i+1][2] = encoder(enc_batch[:,i], enc_s[i][1], enc_s[i][2])
        enc_outputs[:, i, :] = enc_s[i+1][2]

    loss = 0

    for i in range(opt.batch_size):
        dec_s[0][1][i, :] = enc_s[enc_len_batch[i]][1][i, :]
        dec_s[0][2][i, :] = enc_s[enc_len_batch[i]][2][i, :]

    for i in range(dec_max_len):
        dec_s[i+1][1], dec_s[i+1][2] = decoder(dec_batch[:,i], dec_s[i][1], dec_s[i][2])  # FIX IT
        pred = attention_decoder(enc_outputs, dec_s[i+1][2])
        loss += criterion(pred, dec_batch[:,i+1])

    loss = loss / opt.batch_size
    loss.backward()
    torch.nn.utils.clip_grad_value_(encoder.parameters(), opt.grad_clip)
    torch.nn.utils.clip_grad_value_(decoder.parameters(), opt.grad_clip)
    torch.nn.utils.clip_grad_value_(attention_decoder.parameters(), opt.grad_clip)
    encoder_optimizer.step()
    decoder_optimizer.step()
    attention_decoder_optimizer.step()
    return loss


def main(opt):
    # Embedding is not considered in this experiment
    # q_emb_path = '/Users/alvinkennardi/Documents/Master_of_Computing/COMP8755/Embedding/emb_layer_q_split15.pt'
    # f_emb_path = '/Users/alvinkennardi/Documents/Master_of_Computing/COMP8755/Embedding/emb_layer_f_split15.pt'

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    # Load pre-trained model
    checkpoint = torch.load(args.model)
    source_encoder = checkpoint["encoder"]
    source_decoder = checkpoint["decoder"]
    source_attention_decoder = checkpoint["attention_decoder"]

    # get all encoder parameters
    enc_params = source_encoder.state_dict()
    # get all decoder parameters
    dec_params = source_decoder.state_dict()
    # get all attn decoder parameters
    attn_params = source_attention_decoder.state_dict()

    managers = pkl.load(open("{}/map.pkl".format(opt.data_dir), "rb"))
    word_manager, form_manager = managers
    # GPU Settings
    using_gpu = False
    if opt.gpuid > -1:
        using_gpu = True
        torch.cuda.manual_seed(opt.seed)
    # Transfer Learning Settings: LSTM or all transfer
    all_transfer = True
    if opt.attention_transfer == 0:
        all_transfer = False
    # get all initial weight to transfer
    enc_params.pop('embedding.weight')
    dec_params.pop('embedding.weight')
    attn_params.pop('linear_out.weight')
    attn_params.pop('linear_out.bias')

    encoder = RNN(opt, word_manager.vocab_size)
    decoder = RNN(opt, form_manager.vocab_size)
    attention_decoder = AttnUnit(opt, form_manager.vocab_size)

    if using_gpu:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        attention_decoder = attention_decoder.cuda()
    # init parameters
    for name, param in encoder.named_parameters():
        if param.requires_grad:
            init.uniform_(param, -opt.init_weight, opt.init_weight)
    for name, param in decoder.named_parameters():
        if param.requires_grad:
            init.uniform_(param, -opt.init_weight, opt.init_weight)
    for name, param in attention_decoder.named_parameters():
        if param.requires_grad:
            init.uniform_(param, -opt.init_weight, opt.init_weight)

    encoder.load_state_dict(enc_params, strict=False)
    decoder.load_state_dict(dec_params, strict=False)
    if all_transfer:
        attention_decoder.load_state_dict(attn_params, strict=False)

    # encoder.initEmbedding(q_emb_path) # trainable
    # decoder.initEmbedding(f_emb_path) # trainable

    # load data
    train_loader = MiniBatchLoader(opt, 'train', using_gpu)

    if not os.path.exists(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir)

    # start training

    # step = 0
    # epoch = 0
    optim_state = {"learningRate": opt.learning_rate, "alpha": opt.decay_rate}
    # default to RMSprop
    if opt.opt_method == 0:
        print("using RMSprop")
        encoder_optimizer = optim.RMSprop(encoder.parameters(), lr=optim_state["learningRate"],
                                          alpha=optim_state["alpha"])
        decoder_optimizer = optim.RMSprop(decoder.parameters(), lr=optim_state["learningRate"],
                                          alpha=optim_state["alpha"])
        attention_decoder_optimizer = optim.RMSprop(attention_decoder.parameters(), lr=optim_state["learningRate"],
                                                    alpha=optim_state["alpha"])
    criterion = nn.NLLLoss(reduction='sum', ignore_index=0)

    print("Starting training.")
    encoder.train()
    decoder.train()
    attention_decoder.train()
    iterations = opt.max_epochs * train_loader.num_batch
    eval_data = pkl.load(open("{}/test.pkl".format(opt.data_dir), "rb"))
    for i in range(iterations):
        epoch = i // train_loader.num_batch
        start_time = time.time()
        # print("iteration: {}\n".format(i))
        train_loss = eval_training(opt, train_loader, encoder, decoder, attention_decoder, encoder_optimizer,
                                   decoder_optimizer, attention_decoder_optimizer, criterion, using_gpu, form_manager)
        # exponential learning rate decay
        if opt.opt_method == 0:
            if i % train_loader.num_batch == 0 and opt.learning_rate_decay < 1:
                if epoch >= opt.learning_rate_decay_after:
                    decay_factor = opt.learning_rate_decay
                    optim_state["learningRate"] = optim_state["learningRate"] * decay_factor  # decay it
                    for param_group in encoder_optimizer.param_groups:
                        param_group['lr'] = optim_state["learningRate"]
                    for param_group in decoder_optimizer.param_groups:
                        param_group['lr'] = optim_state["learningRate"]
                    for param_group in attention_decoder_optimizer.param_groups:
                        param_group['lr'] = optim_state["learningRate"]

        end_time = time.time()
        if i % opt.print_every == 0:
            print("{}/{}, train_loss = {}, time/batch = {}"
                  .format(i, iterations, train_loss, (end_time - start_time) / 60))

        # TODO: create several checkpoint
        # on last iteration
        if i == iterations - 1:
            checkpoint = {}
            checkpoint["encoder"] = encoder
            checkpoint["decoder"] = decoder
            checkpoint["attention_decoder"] = attention_decoder
            checkpoint["opt"] = opt
            checkpoint["i"] = i
            checkpoint["epoch"] = epoch
            torch.save(checkpoint, "{}/model_seq2seq_attention".format(opt.checkpoint_dir))

        if train_loss != train_loss:
            print('loss is NaN.  This usually indicates a bug.')
            break


if __name__ == "__main__":
    start = time.time()
    main_arg_parser = argparse.ArgumentParser(description="parser")
    main_arg_parser.add_argument('-gpuid', type=int, default=-1, help='which gpu to use. -1 = use CPU')
    main_arg_parser.add_argument('-model', type=str, default='../Src/checkpoint_atis_transfer/model_seq2seq_attention',
                                 help='model checkpoint to use for sampling')
    main_arg_parser.add_argument('-data_dir', type=str, default='../Data/GEO', help='data path')
    main_arg_parser.add_argument('-seed', type=int, default=123, help='torch manual random number generator seed')
    main_arg_parser.add_argument('-checkpoint_dir', type=str, default='checkpoint_atis_geo_transfer',
                                 help='output directory where checkpoints get written')
    main_arg_parser.add_argument('-savefile', type=str, default='save',
                                 help='filename to autosave the checkpont to. Will be inside checkpoint_dir/')
    main_arg_parser.add_argument('-print_every', type=int, default=200,
                                 help='how many steps/minibatches between printing out the loss')
    main_arg_parser.add_argument('-rnn_size', type=int, default=150,
                                 help='size of LSTM internal state')
    main_arg_parser.add_argument('-num_layers', type=int, default=1, help='number of layers in the LSTM')
    main_arg_parser.add_argument('-dropout', type=float, default=0.5,
                                 help='dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
    main_arg_parser.add_argument('-dropoutrec', type=int, default=0,
                                 help='dropout for regularization, used after each c_i. 0 = no dropout')
    main_arg_parser.add_argument('-enc_seq_length', type=int, default=40, help='number of timesteps to unroll for')
    main_arg_parser.add_argument('-dec_seq_length', type=int, default=100,
                                 help='number of timesteps to unroll for')
    main_arg_parser.add_argument('-batch_size', type=int, default=20,
                                 help='number of sequences to train on in parallel')
    main_arg_parser.add_argument('-max_epochs', type=int, default=90,
                                 help='number of full passes through the training data')
    main_arg_parser.add_argument('-opt_method', type=int, default=0, help='optimization method: 0-rmsprop 1-sgd')
    main_arg_parser.add_argument('-learning_rate', type=float, default=0.01, help='learning rate')
    main_arg_parser.add_argument('-init_weight', type=float, default=0.08, help='initailization weight')
    main_arg_parser.add_argument('-learning_rate_decay', type=float, default=0.98, help='learning rate decay')
    main_arg_parser.add_argument('-learning_rate_decay_after', type=int, default=5,
                                 help='in number of epochs, when to start decaying the learning rate')
    main_arg_parser.add_argument('-restart', type=int, default=-1,
                                 help='in number of epochs, when to restart the optimization')
    main_arg_parser.add_argument('-decay_rate', type=float, default=0.95, help='decay rate for rmsprop')
    main_arg_parser.add_argument('-grad_clip', type=int, default=5, help='clip gradients at this value')
    main_arg_parser.add_argument('-attention_transfer', type=int, default=1,
                                 help='0 for only LSTM parameters, 1 for all transfer')

    args = main_arg_parser.parse_args()
    main(args)
    end = time.time()
    print("total time: {} minutes\n".format((end - start) / 60))






    






