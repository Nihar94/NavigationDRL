import argparse
import csv
import os
import sys
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
# import Levenshtein as L
from ctcdecode import CTCBeamDecoder
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.optim as optim
from warpctc_pytorch import CTCLoss
import pandas as pd
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


"""3 stacked BiLSTM layers, each of 256 units. 
This is followed by a 1 Dense layer of 47 (num_classes) units.
Adam optimizer with default learning rate of 1e-3
Decoding is beam search with a beam width of 100."""


class LockedDropout(nn.Module):
    """ LockedDropout applies the same dropout mask to every time step.
    **Thank you** to Sales Force for their initial implementation of :class:`WeightDrop`. Here is
    their `License
    <https://github.com/salesforce/awd-lstm-lm/blob/master/LICENSE>`__.
    Args:
        p (float): Probability of an element in the dropout mask to be zeroed.
    """

    def __init__(self, p=0.5):
        self.p = p
        super().__init__()

    def forward(self, x):
        """
        Args:
            x (:class:`torch.FloatTensor` [batch size, sequence length, rnn hidden size]): Input to
                apply dropout too.
        """
        if not self.training or not self.p:
            return x
        x = x.clone()
        mask = x.new_empty(1, x.size(1), x.size(2), requires_grad=False).bernoulli_(1 - self.p)
        mask = mask.div_(1 - self.p)
        mask = mask.expand_as(x)
        return x * mask

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'p=' + str(self.p) + ')'


class PhonemeModel(nn.Module):

    def __init__(self, n_frequencies=40, hidden_size=256, n_labels=47):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnns = nn.ModuleList()
        self.rnns.append(nn.LSTM(input_size=n_frequencies, hidden_size=hidden_size, bidirectional=True))
        self.rnns.append(nn.LSTM(input_size=2*hidden_size, hidden_size=hidden_size, bidirectional=True))
        self.rnns.append(nn.LSTM(input_size=2*hidden_size, hidden_size=hidden_size, bidirectional=True))
        self.output_layer = nn.Linear(in_features=2*hidden_size, out_features=n_labels)
        self.lockdrop = LockedDropout(p=0.3)

        self.init_layer_weights()

    def forward(self, features):
        """features are given as a packed_padded(n_frames, nu_utt, n_frequencies) and returned like that """
        packed_output = features
        for l in self.rnns:
            packed_output, _ = l(packed_output)
            unpacked_output, output_seq_lengths = pad_packed_sequence(packed_output)
            unpacked_output = self.lockdrop.forward(unpacked_output)
            packed_output = pack_padded_sequence(unpacked_output, output_seq_lengths)

        unpacked_output, output_seq_lengths = pad_packed_sequence(packed_output)
        unpacked_output = self.output_layer(unpacked_output)  # (time_seq, batch, output_logits)
        return unpacked_output, output_seq_lengths

    def init_layer_weights(self):
        # All embedding weights were uniformly initialized in the interval [âˆ’0.1, 0.1] and all other weights were
        # initialized between[âˆ’ 1/âˆšH, 1/âˆšH], where H is the hidden size
        interval = 1/math.sqrt(self.hidden_size)
        lin_int = 0.1
        for layer in self.rnns:
            for name, param in layer.named_parameters():
                nn.init.uniform_(param, a=-interval, b=interval)
        nn.init.uniform_(self.output_layer.weight, a=-lin_int, b=lin_int)


class PhonemeDataLoader(DataLoader):

    def __init__(self, dataset, batch_size, shuffle=True):
        # super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.features = np.load('./wsj0_train.npy', encoding="bytes")
        self.labels = np.load('./wsj0_train_merged_labels.npy', encoding="bytes")
        self.max_feat_len = max([feature.shape[0] for feature in self.features])
        self.amt_features = self.features.shape[0]
        self.amt_batches = self.amt_features//self.batch_size
        print("amt_batches: ", self.amt_features//self.batch_size)

    def __iter__(self):
        if self.shuffle:
            print("je mama")

        random_sequence = np.random.permutation(self.amt_batches*self.batch_size)
        for batch_idx in range(self.amt_batches):
            batch_labels = []
            feature_lens = []
            label_lens = []
            features = []
            for utt_idx in range(self.batch_size):
                idx = random_sequence[batch_idx*self.batch_size+utt_idx]
                padded_feature, label, feature_len, len_label = self.get_item(idx)
                batch_labels.extend(label)
                feature_lens.append(feature_len)
                label_lens.append(len_label)
                features.append(padded_feature)

            max_batch_len = max([len(f) for f in features])
            array_features = np.empty((self.batch_size, max_batch_len, 40))
            for idx in range(self.batch_size):
                feature_len = len(features[idx])
                array_features[idx] = np.pad(features[idx], ((0, max_batch_len - feature_len), (0, 0)), mode='constant')

            labels = torch.autograd.Variable(torch.IntTensor(batch_labels))
            label_lens = torch.IntTensor(label_lens)
            feature_lens = torch.IntTensor(feature_lens)
            result_features = torch.autograd.Variable(torch.from_numpy(array_features))
            yield (result_features.float(), labels, feature_lens, label_lens)

    def get_item(self, item):
        feature = self.features[item]
        feature_len = feature.shape[0]
        label = self.labels[item]
        label += 1
        len_label = label.shape[0]
        label = list(label)
        return feature, label, feature_len, len_label


def shuffle(a, b):
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)


def pack_order(features, seq_lengths):
    ordered_seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    _, unperm_idx = perm_idx.sort(0)
    ordered_batch = features[perm_idx, :, :]  # batch first
    ordered_batch = ordered_batch.permute(1, 0, 2)
    packed_padded_ordered_batch = pack_padded_sequence(ordered_batch, ordered_seq_lengths.cpu().numpy())
    return packed_padded_ordered_batch.cuda(), unperm_idx.cuda()   # seq first returned


def train(model):
    model.cuda()
    model.train()
    print("loading train and dev data...")
    eval_set = PhonemeTrainDataset(purpose='dev')
    train_loader = PhonemeDataLoader(dataset=0, shuffle=True, batch_size=BATCH_SIZE)
    print("data loaded")
    optimizer = optim.Adam(model.parameters())
    criterion = CTCLoss(size_average=True)
    torch.save(model.state_dict(), "./checkpoint" + str(99))
    print("test model saved")

    for e in range(N_EPOCHS):

        print("EPOCH started", e)
        epoch_loss = 0
        amt_batches = 0
        for index, (data_batch, label_batch, seq_lens, label_lens) in enumerate(train_loader):
            optimizer.zero_grad()
            data_batch = data_batch.cuda()
            label_batch = label_batch.cuda()
            label_lens = label_lens.int().cpu()

            prepped_data_batch, unperm_idx = pack_order(data_batch, seq_lengths=seq_lens)
            acts, act_lens = model(prepped_data_batch)
            acts = acts[:, unperm_idx, :]  # seq first
            act_lens = act_lens[unperm_idx]

            labels = label_batch.view(-1).int().cpu()
            act_lens = act_lens.int().cpu()
            loss = criterion.forward(acts, labels, act_lens, label_lens)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if index % 10 == 9:
                print("batch number:", index, "loss:", epoch_loss/index)
            amt_batches = index

        torch.save(model.state_dict(), "./checkpoint" + str(e))
        print("model saved")
        print("EPOCH", e, "ENDED, Loss: {}".format(epoch_loss / amt_batches))


def predict(model):

    test_dataset = PhonemeTestDataset()
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    label_map = [' '] + PHONEME_MAP
    decoder = CTCBeamDecoder(labels=label_map, blank_id=0)
    model.eval()
    model.cuda()

    prediction = []

    for index, utt in enumerate(test_loader):
        utt = utt.cuda()
        pred, seq_lens = model(utt)  # pred = (seq, 1, logits)
        logits = torch.transpose(pred, 0, 1)  # (1, seq, logits)
        probs = F.softmax(logits, dim=2).data.cpu()
        output, scores, timesteps, out_seq_len = decoder.decode(probs=probs, seq_lens=seq_lens)
        for i in range(output.size(0)):  # output_size(0)==1 for now
            chrs = [label_map[o.item()] for o in output[i, 0, :out_seq_len[i, 0]]]
            chrs = ''.join(chrs)
            prediction.append(chrs)
        if index % 10 == 0:
            print("test_index: ", index, "/523")

    df = pd.DataFrame(prediction)
    df.to_csv('./handin.csv')
    print("csv file is ready to roll")


PHONEME_MAP = [
    '_',  # "+BREATH+"
    '+',  # "+COUGH+"
    '~',  # "+NOISE+"
    '!',  # "+SMACK+"
    '-',  # "+UH+"
    '@',  # "+UM+"
    'a',  # "AA"
    'A',  # "AE"
    'h',  # "AH"
    'o',  # "AO"
    'w',  # "AW"
    'y',  # "AY"
    'b',  # "B"
    'c',  # "CH"
    'd',  # "D"
    'D',  # "DH"
    'e',  # "EH"
    'r',  # "ER"
    'E',  # "EY"
    'f',  # "F"
    'g',  # "G"
    'H',  # "HH"
    'i',  # "IH"
    'I',  # "IY"
    'j',  # "JH"
    'k',  # "K"
    'l',  # "L"
    'm',  # "M"
    'n',  # "N"
    'G',  # "NG"
    'O',  # "OW"
    'Y',  # "OY"
    'p',  # "P"
    'R',  # "R"
    's',  # "S"
    'S',  # "SH"
    '.',  # "SIL"
    't',  # "T"
    'T',  # "TH"
    'u',  # "UH"
    'U',  # "UW"
    'v',  # "V"
    'W',  # "W"
    '?',  # "Y"
    'z',  # "Z"
    'Z',  # "ZH"
]


class ER:

    def __init__(self):
        self.label_map = [' '] + DIGITS_MAP
        self.decoder = CTCBeamDecoder(
            labels=self.label_map,
            blank_id=0
        )

    def __call__(self, prediction, target):
        return self.forward(prediction, target)

    def forward(self, prediction, target):
        logits = prediction[0]
        feature_lengths = prediction[1].int()
        labels = target + 1
        logits = torch.transpose(logits, 0, 1)
        logits = logits.cpu()
        probs = F.softmax(logits, dim=2)
        output, scores, timesteps, out_seq_len = self.decoder.decode(probs=probs, seq_lens=feature_lengths)

        pos = 0
        ls = 0.
        for i in range(output.size(0)):
            pred = "".join(self.label_map[o] for o in output[i, 0, :out_seq_len[i, 0]])
            true = "".join(self.label_map[l] for l in labels[pos:pos + 10])
            #print("Pred: {}, True: {}".format(pred, true))
            pos += 10
            ls += L.distance(pred, true)
        assert pos == labels.size(0)
        return ls / output.size(0)


def run_eval(model, eval_set):
    error_rate_op = ER()
    model.cuda()
    model.eval()
    # eval_set = PhonemeTrainDataset(purpose=dev)
    loader = DataLoader(eval_set, shuffle=False, batch_size=1)
    predictions = []
    feature_lengths = []
    labels = []
    for idx, (data_batch, labels_batch, label_lens) in loader:
        data_batch = data_batch.cuda()
        predictions_batch, feature_lengths_batch = model(data_batch)
        predictions.append(predictions_batch.to("cpu"))
        feature_lengths.append(feature_lengths_batch.to("cpu"))
        labels.append(labels_batch.cpu())
    predictions = torch.cat(predictions, dim=1)
    labels = torch.cat(labels, dim=0)
    feature_lengths = torch.cat(feature_lengths, dim=0)
    error = error_rate_op((predictions, feature_lengths), labels.view(-1))

    return error


BATCH_SIZE = 128
N_EPOCHS = 10
net = PhonemeModel()
train(net)
predict(net)