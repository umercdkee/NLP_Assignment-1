#parent class for both tokenizers to avoid repeating methods
import unicodedata
from collections import Counter

class Tokenizer:
    def __init__(self, file):
        with open(file, encoding="utf-8") as f:
            self.text = f.read()
            self.text = unicodedata.normalize("NFC", self.text)
            self.text = self.text.lower()
            self.text = "".join(
                f" {char} " if unicodedata.category(char).startswith("P") else char
                for char in self.text
            )
            self.text = " ".join(self.text.split())


    def output_tokens(self, vocab, filename):
            with open(filename, "w", encoding="utf-8") as f:
                for token in vocab:
                    f.write(token + "\n")

#WordPieceTokenizer class inherits from Tokenizer and implements the tokenize method
class WordPieceTokenizer(Tokenizer):
    def __init__(self, file, vocab_size=100):
        super().__init__(file)
        self.tokens = self.tokenize()
        self.vocab_size = vocab_size

    def tokenize(self):
        tokens = []
        for word in self.text.split():
            word_tokens = []
            for i in range(len(word)):
                if i == 0:
                    word_tokens.append(word[i])
                else:
                    word_tokens.append("##" + word[i])
            tokens.append(word_tokens)  
        return tokens

    def vocabulary_builder(self):
        
        vocab = {token for word in self.tokens for token in word}

        if self.vocab_size <= len(vocab):
            print(f"vocab_size ({self.vocab_size}) is too small. "
                f"Initial vocab already has {len(vocab)} tokens.")
            return vocab

        while len(vocab) < self.vocab_size:

            token_freq = Counter(
                        token
                        for word in self.tokens
                        for token in word
                    )
            pair_freq = Counter(
                        (word[i], word[i + 1])
                        for word in self.tokens
                        for i in range(len(word) - 1)
                    )

            if not pair_freq:
                break

            best_pair = max(
                pair_freq,
                key=lambda pair:
                    pair_freq[pair]
                    / (token_freq[pair[0]] * token_freq[pair[1]])
            )
            new_token = (best_pair[0]+ best_pair[1].replace("##", ""))

            for word in self.tokens:
                i = 0
                while i < len(word) - 1:
                    if (word[i] == best_pair[0] and word[i + 1] == best_pair[1]):
                        word[i] = new_token
                        del word[i + 1]
                    i += 1

            vocab.add(new_token)
        return vocab

class BPETokenizer(Tokenizer):
    def __init__(self, file, vocab_size=100):
        super().__init__(file)
        self.tokens = self.tokenize()
        self.vocab_size = vocab_size

    def tokenize(self):
        tokens = []
        for word in self.text.split():
            tokens.append(list(word))
        return tokens

    def vocabulary_builder(self):
        vocab = set(token for word in self.tokens for token in word)
        while len(vocab) < self.vocab_size:
            pair_freq = Counter(
                        (word[i], word[i + 1])
                        for word in self.tokens
                        for i in range(len(word) - 1)
                    )
            if not pair_freq:
                break

            best_pair = max(
                pair_freq,
                key=lambda pair: pair_freq[pair]
            )
            new_token = best_pair[0] + best_pair[1]

            for word in self.tokens:
                i = 0
                while i < len(word) - 1:
                        if word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                            word[i] = new_token
                            del word[i + 1]
                        i += 1
            vocab.add(new_token)

        return vocab

vocab_size = int(input("Enter the desired vocabulary size for WordPiece: "))
wordpiece_tokenizer = WordPieceTokenizer("corpus.txt", vocab_size=vocab_size)
vocab = wordpiece_tokenizer.vocabulary_builder()
wordpiece_tokenizer.output_tokens(vocab, "wordpiece_vocab.txt")

vocab_size = int(input("Enter the desired vocabulary size for BPE: "))
bpe_tokenizer = BPETokenizer("corpus.txt", vocab_size=vocab_size)
vocab = bpe_tokenizer.vocabulary_builder()
bpe_tokenizer.output_tokens(vocab, "bpe_vocab.txt")
