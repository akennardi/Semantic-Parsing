from nltk.tokenize import MWETokenizer


class Lang:
    def __init__(self):
        self.w2i = {}  # word to index mapping
        self.w2c = {}  # word to count mapping
        self.i2w = {}  # index to word mapping
        self.n_words = 0

    def add_sentence(self, sentence, phrase_list):
        sentence = sentence.lower()
        tokenizer = MWETokenizer(phrase_list)
        tokens = tokenizer.tokenize(sentence.split())
        for word in tokens:
            self.add_word(word)

    def add_word(self, word):
        if word not in self.w2i:
            self.w2i[word] = self.n_words
            self.w2c[word] = 1
            self.i2w[self.n_words] = word
            self.n_words += 1
        else:
            self.w2c[word] += 1
