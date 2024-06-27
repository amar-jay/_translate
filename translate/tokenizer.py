from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Digits, Punctuation, WhitespaceSplit
from tokenizers.normalizers import NFD
from tokenizers.normalizers import Lowercase
import tiktoken
import os

class MyTokenizer():
    def __init__(self, tokenizer_path, data_path = []):
        for path in data_path:
            if not os.path.exists(path):
                raise Exception("Data path does not exist")

        self.tokenizer_path = tokenizer_path
        self.data_path = data_path
        self.tokenizer = None
        self.normalizer = normalizers.Sequence([NFD(), Lowercase()]) # type: ignore
        self.pre_tokenizer = pre_tokenizers.Sequence([WhitespaceSplit(), Punctuation(), Digits(individual_digits=True)])
    def train(self):
        if os.path.exists(tokenizer_path):
            print("tokenizer file exists")
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
        else:
            print("tokenizer file does not exist")
            self.force_train()
    def force_train(self):
        self.tokenizer = Tokenizer(BPE())
        self.tokenizer.normalizer = self.normalizer # type: ignore
        self.tokenizer.pre_tokenizer = self.pre_tokenizer # type: ignore
        self.tokenizer.train(self.data_path, trainer=BpeTrainer())
        self.tokenizer.save(self.tokenizer_path)
    def encode(self, text):
        if self.tokenizer is None:
            raise Exception("Tokenizer is not trained")

        return self.tokenizer.encode(text)
    def with_gpt(self):
        tokenizer = tiktoken.get_encoding("gpt2")
        return {
            'encode': lambda s: tokenizer.encode(s, allowed_special={"<|endoftext|>"}),
            'decode': lambda t: tokenizer.decode(t)
        }

    def decode(self, ids):
        if self.tokenizer is None:
            raise Exception("Tokenizer is not trained")
        return self.tokenizer.decode(ids)


if __name__ == "__main__":
    tokenizer_path = os.path.join(os.path.dirname(__file__), "..", "data", "parallel_wiki.json")
    eng_data_path = os.path.join(os.path.dirname(__file__), "..", "data", "eng_parallel_corpus.txt")
    tur_data_path = os.path.join(os.path.dirname(__file__), "..", "data", "tur_parallel_corpus.txt")
    tokenizer = MyTokenizer(tokenizer_path, [eng_data_path, tur_data_path])
    tokenizer.train()
    encoded = tokenizer.encode("The Structure of a Single-Phase Transformer: A transformer is a device that transfers electrical energy from one voltage level to another (without changing the frequency) through electrical induction")
    print("ids: %s\ntokens: %s" % (encoded.ids, encoded.tokens))
    decoded = tokenizer.decode(encoded.ids)
    print("decoded: %s\n\n" % decoded)
    print("-"*20)
    encoded = tokenizer.encode("Bir Fazlı Trafonun Yapısı Transformatör, elektrik enerjisini bir gerilim seviyesinden başka bir gerilim seviyesine (frekansını değiştirmeden) elektriksel endüksiyon ile aktaran bir cihazdır.")
    print("ids: %s\ntokens: %s" % (encoded.ids, encoded.tokens))
    decoded = tokenizer.decode(encoded.ids)
    print("decoded: %s\n\n" % decoded)
    print("-"*20)
    print("Vocab size:\t ", tokenizer.tokenizer.get_vocab_size())