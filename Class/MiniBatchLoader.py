from random import randint
import pickle as pkl
import torch


torch.manual_seed(1)


class MiniBatchLoader:
    def __init__(self, opt, mode, using_gpu):
        data = pkl.load( open("{}/{}.pkl".format(opt.data_dir, mode), "rb" ) )
        data = sorted(data, key=lambda d: len(d[0])) # sort all data based on the length of the question
        if len(data) % opt.batch_size != 0:
            n = len(data)
            for i in range(len(data)%opt.batch_size):
                data.insert(n-i-1, data[n-i-1])
        self.enc_batch_list = []
        self.enc_len_batch_list = []
        self.dec_batch_list = []
        p = 0  # batch ordering, i.e. if batch size =20, it will be 0, 20, 40, 60, ... in each iteration

        while p + opt.batch_size <= len(data):
            # assumption: data is sorted based on the length of question. The last question on each batch is the longest
            max_len = len(data[p + opt.batch_size - 1][0])
            m_text = torch.zeros((opt.batch_size, max_len + 2), dtype=torch.long)
            if using_gpu:
                m_text = m_text.cuda()
            enc_len_list = []
            # add <S>
            m_text[:,0] = 0
            for i in range(opt.batch_size):
                w_list = data[p + i][0]
                # reversed order
                for j in range(len(w_list)):
                    # print(max_len+2)
                    m_text[i][j+1] = w_list[len(w_list) - j -1]
                    # m_text[i][j+1] = w_list[j]
                # -- add <E> (for encoder, we need dummy <E> at the end)
                for j in range(len(w_list)+1, max_len+2):
                    m_text[i][j] = 1
                enc_len_list.append(len(w_list)+2)
            # print (m_text)
            self.enc_batch_list.append(m_text)
            self.enc_len_batch_list.append(enc_len_list)
            # build decoder matrix
            max_len = -1
            for i in range(opt.batch_size):
                w_list = data[p+i][1]
                if len(w_list) > max_len:
                    max_len = len(w_list)
            m_text = torch.zeros((opt.batch_size, max_len + 2), dtype=torch.long)
            if using_gpu:
                m_text = m_text.cuda()
            # add <S>
            m_text[:, 0] = 0
            for i in range(opt.batch_size):
                w_list = data[p+i][1]
                for j in range(len(w_list)):
                    m_text[i][j+1] = w_list[j]
                # add <E>
                m_text[i][len(w_list)+1] = 1
            self.dec_batch_list.append(m_text)
            p += opt.batch_size

        self.num_batch = len(self.enc_batch_list)
        assert(len(self.enc_batch_list) == len(self.dec_batch_list))

    def random_batch(self):
        p = randint(0, self.num_batch-1)
        return self.enc_batch_list[p], self.enc_len_batch_list[p], self.dec_batch_list[p]

    def all_batch(self):
        r = []
        for p in range(self.num_batch):
            r.append([self.enc_batch_list[p], self.enc_len_batch_list[p], self.dec_batch_list[p]])
        return r
