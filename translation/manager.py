import torch
import torch.nn as nn
from torch import Tensor

from translation.decoder import triu_mask
from translation.model import Model

Optimizer = torch.optim.Optimizer
LRScheduler = torch.optim.lr_scheduler.LRScheduler


class Vocab:
    def __init__(self):
        self.num_to_word = ['<UNK>', '<BOS>', '<EOS>', '<PAD>']
        self.word_to_num = {x: i for i, x in enumerate(self.num_to_word)}

        self.UNK = self.word_to_num['<UNK>']
        self.BOS = self.word_to_num['<BOS>']
        self.EOS = self.word_to_num['<EOS>']
        self.PAD = self.word_to_num['<PAD>']

    def add(self, word: str):
        if word not in self.word_to_num:
            self.word_to_num[word] = self.size()
            self.num_to_word.append(word)

    def numberize(self, words: list[str]) -> list[int]:
        return [self.word_to_num[word] if word in self.word_to_num else self.UNK for word in words]

    def denumberize(self, nums: list[int]) -> list[str]:
        try:
            start = len(nums) - nums[::-1].index(self.BOS)
        except ValueError:
            start = 0
        try:
            end = nums.index(self.EOS)
        except ValueError:
            end = len(nums)
        return [self.num_to_word[num] for num in nums[start:end]]

    def size(self) -> int:
        return len(self.num_to_word)


class Batch:
    def __init__(
        self,
        src_nums: Tensor,
        tgt_nums: Tensor,
        kernel_size: int,
        ignore_index: int,
        device: str = 'cpu',
    ):
        self._src_nums = src_nums
        self._tgt_nums = tgt_nums
        self.kernel_size = kernel_size
        self.ignore_index = ignore_index
        self.device = device

    @property
    def src_nums(self) -> Tensor:
        return self._src_nums.to(self.device)

    @property
    def tgt_nums(self) -> Tensor:
        return self._tgt_nums.to(self.device)

    @property
    def src_mask(self) -> Tensor:
        return (self.src_nums != self.ignore_index).unsqueeze(-2)

    @property
    def tgt_mask(self) -> Tensor:
        # return triu_mask(self.tgt_nums.size(-1), device=self.device)
        N = self.kernel_size
        return triu_mask(self.tgt_nums[:, :-N].size(-1), device=self.device)

    def length(self) -> int:
        N = self.kernel_size
        return int((self.tgt_nums[:, N:] != self.ignore_index).sum())

    def size(self) -> int:
        return self._src_nums.size(0)


# class Tokenizer:
#     def __init__(self, src_lang: str, tgt_lang: str | None = None, sw_model: Any | None = None):
#         self.src_lang = src_lang
#         self.tgt_lang = tgt_lang
#         self.normalizer = MosesPunctNormalizer(src_lang)
#         self.tokenizer = MosesTokenizer(src_lang)
#         lang = tgt_lang if tgt_lang else src_lang
#         self.detokenizer = MosesDetokenizer(lang)
#         self.sw_model = sw_model

#     def tokenize(self, text: str) -> list[str]:
#         text = self.normalizer.normalize(text)
#         tokens = self.tokenizer.tokenize(text, escape=False)
#         if self.sw_model is None:
#             return tokens
#         if isinstance(self.sw_model, BPE):
#             return self.sw_model.process_line(' '.join(tokens)).split()
#         return self.sw_model.encode_as_pieces(' '.join(tokens))

#     def detokenize(self, tokens: list[str]) -> str:
#         if self.sw_model:
#             if isinstance(self.sw_model, BPE):
#                 text = re.sub('(@@ )|(@@ ?$)', '', ' '.join(tokens))
#             else:
#                 # text = ''.join(tokens).replace('▁', ' ').strip()
#                 text = self.sw_model.decode(tokens)
#         return self.detokenizer.detokenize(text.split())


class Manager:
    embed_dim: int
    ff_dim: int
    num_heads: int
    dropout: float
    num_layers: int
    max_epochs: int
    lr: float
    patience: int
    decay_factor: float
    min_lr: float
    max_patience: int
    label_smoothing: float
    clip_grad: float
    batch_size: int
    max_length: int
    beam_size: int
    kernel_size: int

    def __init__(
        self,
        config: dict,
        device: str,
        # src_lang: str,
        # tgt_lang: str,
        model_file: str,
        sw_vocab_file: str,
        # sw_model_file: str,
    ):
        self.config = config
        self.device = device
        # self.src_lang = src_lang
        # self.tgt_lang = tgt_lang
        self._model_name = model_file

        for option, value in config.items():
            self.__setattr__(option, value)

        with open(sw_vocab_file) as sw_vocab_f:
            self.vocab = Vocab()
            for line in sw_vocab_f.readlines():
                self.vocab.add(line.split()[0])
        # with open(sw_model_file) as sw_model_f:
        #     header = sw_model_f.readline()
        # if header.startswith('#version'):
        #     with open(sw_model_file) as sw_model_f:
        #         self.sw_model = BPE(sw_model_f)
        # else:
        #     self.sw_model = spm.SentencePieceProcessor(sw_model_file)

        self.model = Model(
            self.vocab.size(),
            self.embed_dim,
            self.ff_dim,
            self.num_heads,
            self.dropout,
            self.num_layers,
            self.kernel_size,
        ).to(device)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        self.model.apply(init_weights)

    def save_model(
        self, train_state: tuple[int, float], optimizer: Optimizer, scheduler: LRScheduler
    ):  # train_state: (Final Epoch, Best Loss)
        torch.save(
            {
                'config': self.config,
                # 'src_lang': self.src_lang,
                # 'tgt_lang': self.tgt_lang,
                'optimizer': optimizer.state_dict,
                'scheduler': scheduler.state_dict,
                'state_dict': self.model.state_dict(),
                'train_state': train_state,
            },
            self._model_name,
        )

    def batch_data(self, data: list) -> list[Batch]:
        batched_data = []

        data.sort(key=lambda x: (len(x[0]), len(x[1])), reverse=True)

        i = batch_size = 0
        while (i := i + batch_size) < len(data):
            src_len, tgt_len = len(data[i][0]), len(data[i][1])

            while True:
                # seq_len = math.ceil(max(src_len, tgt_len) / 8) * 8
                # batch_size = max(self.batch_size // (seq_len * 8) * 8, 1)
                batch_size = max(self.batch_size // max(src_len, tgt_len), 1)

                src_batch, tgt_batch = zip(*data[i : (i + batch_size)])
                # src_len = math.ceil(max(len(src_words) for src_words in src_batch) / 8) * 8
                # tgt_len = math.ceil(max(len(tgt_words) for tgt_words in tgt_batch) / 8) * 8
                src_len = max(len(src_words) for src_words in src_batch)
                tgt_len = max(len(tgt_words) for tgt_words in tgt_batch)

                if batch_size * max(src_len, tgt_len) <= self.batch_size:
                    break
            assert batch_size > 0

            src_nums = torch.stack(
                [
                    nn.functional.pad(
                        torch.tensor(self.vocab.numberize(src_words)),
                        (0, src_len - len(src_words)),
                        value=self.vocab.PAD,
                    )
                    for src_words in src_batch
                ]
            )
            tgt_nums = torch.stack(
                [
                    nn.functional.pad(
                        torch.tensor(self.vocab.numberize(tgt_words)),
                        (0, tgt_len - len(tgt_words)),
                        value=self.vocab.PAD,
                    )
                    for tgt_words in tgt_batch
                ]
            )

            batched_data.append(
                Batch(src_nums, tgt_nums, self.kernel_size, self.vocab.PAD, self.device)
            )

        return batched_data

    def load_data(self, data_file: str) -> list[Batch]:
        data = []
        with open(data_file) as data_f:
            N = self.kernel_size
            for line in data_f.readlines():
                src_line, tgt_line = line.split('\t')
                src_words = N * ['<BOS>'] + src_line.split() + N * ['<EOS>']
                tgt_words = N * ['<BOS>'] + tgt_line.split() + N * ['<EOS>']
                data.append((src_words, tgt_words))
        return self.batch_data(data)
