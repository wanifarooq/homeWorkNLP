# *****************************************************************************************************************************
# I------------------------- have added this as an extra.--------------------------------------------------------------------
# this file contains all of the implementation of the bidirectional network the code quality will not be good because i haven't
# optimized it as  this method never gave me the better accuracy then first two methods.
# Everything is same as that of the second approach but here i am using bidirectional LSTM and the collate function
# is different return the five outputs. Concatenated word embedded indies of the two sentences, actual sequence length
# in first sentence , actual sequence length in second sentence , point of the concatenation , and the target label optional
# and in model part i am taking four outputs from the bidirectional LSTM and then doing different operations on them
# like taking first 10 values of each output for the forward context and next ten for the backward context, then taking
# the euclidean distance between the two vectors and passing them for the rest of the network
# *******************************************************************************************************************************

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict
from tqdm.notebook import tqdm
from typing import *
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
from tqdm.auto import tqdm
import os
import json
import re
import distutils

def rnn_collate_fn(data_elements: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    X = [de[0] for de in data_elements]
    X_lengths = torch.tensor([x.size(0) for x in X], dtype=torch.long)
    X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=0)
    offset = X.shape[1] + 1
    mid = torch.zeros(X.shape[0]).type(torch.long)
    mid[:] = offset
    X1 = [de[1] for de in data_elements]
    X1_lengths = torch.tensor([(x.size(0) + offset) for x in X1], dtype=torch.long)
    X1 = torch.nn.utils.rnn.pad_sequence(X1, batch_first=True, padding_value=0)
    y = [de[2] for de in data_elements]
    y = torch.tensor(y).type(torch.float32)
    zeros = torch.zeros([X1.shape[0], 1]).type(torch.long)
    final = torch.cat([X, zeros, X1], dim=1)

    return final, y, X_lengths, X1_lengths, mid



class SecondNetwork(torch.nn.Module):
    def __init__(self, vectors: torch.Tensor, n_hidden: int, device) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(vectors)
        self.rnn = torch.nn.LSTM(input_size=vectors.size(1), hidden_size=n_hidden, num_layers=1, batch_first=True,
                                 bidirectional=True)
        self.drop = nn.Dropout(p=0.5)
        self.lin1 = torch.nn.Linear(2 * n_hidden, 1)
#         self.lin2 = torch.nn.Linear(n_hidden, 1)
        self.loss_fn = torch.nn.BCELoss()
        self.device = device
        self.n_hidden = n_hidden

    def forward(self, X: torch.Tensor, X_length: torch.Tensor, X1_length: torch.Tensor,mid: torch.Tensor,
                y: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        embedding_out = self.embedding(X)
        recurrent_out = self.rnn(embedding_out)[0]
        batch_size, seq_len, hidden_size = recurrent_out.shape
        flattened_out = recurrent_out.reshape(-1, hidden_size)

        first_word_relative_indices = X_length - 1
        second_word_relative_indices = X1_length - 1
        sequences_offsets = torch.arange(batch_size, device=self.device) * seq_len

        first_indices = sequences_offsets + first_word_relative_indices
        second_indices = sequences_offsets + second_word_relative_indices

        f1 = flattened_out[first_indices]
        f2= flattened_out[second_indices]
        r1 = flattened_out[sequences_offsets]
        r2 = flattened_out[mid]

#         f1_forward= f1[:,:self.n_hidden]-r1[:,:self.n_hidden]
#         f2_forward= f2[:,:self.n_hidden]-r2[:,:self.n_hidden]
#         r1_reverse= r1[:, self.n_hidden:]-f1[:, self.n_hidden:]
#         r2_reverse= r2[:, self.n_hidden:]-f2[:, self.n_hidden:]
#         first_vector = torch.cat((f1_forward,r1_reverse),1)
#         second_vector = torch.cat((f2_forward,r2_reverse),1)
#         f_vector = (first_vector - second_vector)**2


        first_vector = f1[:,:self.n_hidden]
        second_vector = f2[:,:self.n_hidden]
        reverse_firstVector = r1[:, self.n_hidden:]
        reverse_secondVector = r2[:, self.n_hidden:]
        full_first_vector = torch.cat((first_vector, reverse_firstVector), 1)
        full_second_vector = torch.cat((second_vector,reverse_secondVector),1)
        f_vector = (full_first_vector - full_second_vector)**2



#         f_vector = self.drop(f_vector)
        out = self.lin1(f_vector).squeeze(1)
#         out = torch.relu(out)
#         out = self.lin2(out).squeeze(1)
        out = torch.sigmoid(out)
        result = {'pred': out}
        if y is not None:
            loss = self.loss_fn(out, y)
            result['loss'] = loss
        return result

@torch.no_grad()
def evaluate_accuracy(model: nn.Module, dataloader: DataLoader):
    correct_predictions = 0
    num_predictions = 0
    for x, y, z1, z2 ,mid in dataloader:
        outputs = model(x.to(device), z1.to(device), z2.to(device),mid.to(device))
        predictions = outputs['pred']
        predictions = torch.round(predictions)
        correct_predictions += (predictions == y.to(device)).sum()
        num_predictions += predictions.shape[0]

    accuracy = correct_predictions / num_predictions
    return accuracy



class Trainer():
    def __init__(self, model, optimizer, device, evaluate_accuracy: Callable = None):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.accuracy = evaluate_accuracy
        self.model.train()
        self.model.to(self.device)

    def train(self, train_dataset, output_folder, test_dataset, epochs=5, interval=5):

        train_history = []
        print(epochs)
        for epoch in range(epochs):
            epoch_loss = 0.0
            len_train = 0

            # each element (sample) in train_dataset is a batch
            for step, sample in tqdm(enumerate(train_dataset), desc="Batch", leave=False):
                inputs = sample[0].to(self.device)
                targets = sample[1].to(self.device)
                x_length = sample[2].to(self.device)
                x1_length = sample[3].to(self.device)
                mid=sample[4].to(self.device)
                #                 print(targets)
                batch_out = model(inputs, x_length, x1_length,mid,targets)
                loss = batch_out['loss']

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                len_train += 1
            if (epoch % interval) == 0:
                avg_epoch_loss = epoch_loss / len_train
                train_history.append(avg_epoch_loss)
                acc = self.accuracy(model, test_dataset)
                print('Epoch: {} avg loss = {:0.4f} avg acc = {:0.4f}'.format(epoch, avg_epoch_loss, acc))

                torch.save(self.model.state_dict(),
                           os.path.join(output_folder, 'Rnnstate_{}.pt'.format(epoch)))  # save the model state
                # assert self.accuracy is not None

        return {'train_history': train_history}


