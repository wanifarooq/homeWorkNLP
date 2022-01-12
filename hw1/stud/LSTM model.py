# *******************************************************************************************************************
# this file is the implementation of the LSTM and the code quality is better and optimized, i haven't choose it as my
# method because it accuracy was not better than first method though it takes lesser epochs to reach maximum accuracy.
# Added for the reference only for the actual implementation of the first method refer to implementation.py
# *******************************************************************************************************************


import numpy as np
from tqdm.notebook import tqdm
from typing import *
from collections import Counter, defaultdict
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import os
import json
import re
import distutils
from typing import List, Dict

# Here the data processing is as of the first model till collate function where difference occurs
# The predict function is already implemented in student class and here it is just for testing
class Model:

    def predict(self, sentence_pairs: List[Dict]) -> List[Dict]:


        raise NotImplementedError

# This is again the data_loader class but here rather than embedding glove vectors for sentence i am embedding the
# indices of the vectors.
# The main difference is in the collate function which is described below
class RnnDataload():

    def __init__(self, path_pretrained_weights: str):
        super().__init__()
        self.wordVector_index ,self.vectors= self.loadPreTrainedWeights(path_pretrained_weights)
        self.rnn_collate_fn =rnn_collate_fn

    def loadPreTrainedWeights(self,path_pretrained_weights):
        f = open(path_pretrained_weights,'r' ,encoding="utf8")

        wordVector_index = dict()
        vectors = []
        vectors.append(torch.zeros([100]))
        vectors.append(torch.rand(100))
        for line in f:
            splitLines = line.split()
            word = splitLines[0]
            vector = np.array([float(value) for value in splitLines[1:]])
            wordVector_index[word] = len(vectors)
            vectors.append(torch.Tensor(vector))
        wordVector_index = defaultdict(lambda: 1, wordVector_index)  # default dict returns 1 (unk token) when unknown word
        vectors = torch.stack(vectors)
        return wordVector_index , vectors

    def sentenceToIndices(self,statememt, pattern='\W'):
        return torch.tensor([self.wordVector_index[w] for w in [word for word in re.split(pattern, statememt.lower()) if word]])

    def get_data(self,data ,operation,batch = None):
        sample_data=[]

        for parse in data:
            sentence1 = parse['sentence1']
            sentence2 = parse['sentence2']
            word1 = sentence1[int(parse['start1']):int(parse['end1'])]
            word2 = sentence2[int(parse['start2']):int(parse['end2'])]
            if (word1.lower()) != (word2.lower()):
                sentence2 = "".join((sentence2[:int(parse['start2'])],word1,sentence2[int(parse['end2']):]))
            sentence1 = self.sentenceToIndices(sentence1)
            sentence2 = self.sentenceToIndices(sentence2)
            if None not in [sentence1,sentence2]:
                if (operation=='train' or operation=='dev' ):
                    label = (1 if distutils.util.strtobool(parse['label']) else 0)
                    sample_data.append( (sentence1, sentence2,label))
                else:
                    sample_data.append( (sentence1,sentence2))
        return DataLoader(sample_data, batch_size=batch, collate_fn=self.rnn_collate_fn)

# This is very typical function which accepts the indices vector for each of the sentence
# Infact it accepts the list of the tuple of sentence1 word embedded indices , sentence2 word embedded indices and the
# optional target label which will be used only during the training and validation but not for testing.
# it returns the batch of equal length sequences of two sentences concatenated by the zero vector and the actual length
# of sequences present in each sentence.
def rnn_collate_fn(data_elements: List[Tuple[torch.Tensor, torch.Tensor,Optional[torch.Tensor] ]]) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    X = [de[0] for de in data_elements]
    X_lengths = torch.tensor([x.size(0) for x in X], dtype=torch.long)
    X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=0)
    offset = X.shape[1] + 1
    X1 = [de[1] for de in data_elements]
    X1_lengths = torch.tensor([(x.size(0) + offset) for x in X1], dtype=torch.long)
    X1 = torch.nn.utils.rnn.pad_sequence(X1, batch_first=True, padding_value=0)
    zeros = torch.zeros([X1.shape[0], 1]).type(torch.long)
    final = torch.cat([X, zeros, X1], dim=1)
    try:
        y = [de[2] for de in data_elements]
        y = torch.tensor(y).type(torch.float32)
        return final, X_lengths, X1_lengths,y
    except IndexError:
        pass
    return final, X_lengths, X1_lengths
#This is the actual network design or architecture for the use of the LSTM.
#Most of the model architecture is self explanatory except the thing that i am taking the two outputs for LSTM at the
#sequence length of the first sentence and the sequence length of the second sentence
#taking the euclidean distance of the two outputs before sending them to the further network
#Rest of the code is self explanatory
class SecondNetwork(torch.nn.Module):
    def __init__(self, vectors: torch.Tensor, n_hidden: int, device) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(vectors)
        self.rnn = torch.nn.LSTM(input_size=vectors.size(1), hidden_size=n_hidden, num_layers=1,batch_first=True)
        self.lin1 = torch.nn.Linear(n_hidden, n_hidden)
        self.lin2 = torch.nn.Linear(n_hidden, 1)
        self.loss_fn = torch.nn.BCELoss()
        self.device = device

    def forward(self, X: torch.Tensor, X_length: torch.Tensor, X1_length: torch.Tensor,
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

        first_vector = flattened_out[first_indices]
        second_vector = flattened_out[second_indices]
        f_vector = (second_vector - first_vector)**2

        out = self.lin1(f_vector)
        out = torch.relu(out)
        out = self.lin2(out).squeeze(1)
        out = torch.sigmoid(out)
        result = {'pred': out}
        if y is not None:
            loss = self.loss_fn(out, y)
            result['loss'] = loss
        return result

@torch.no_grad()
def evaluate_accuracy_rnn(model: nn.Module, dataloader: DataLoader,device):
    correct_predictions = 0
    num_predictions = 0
    for x,z1, z2,y in dataloader:
        outputs = model(x.to(device), z1.to(device), z2.to(device))
        predictions = outputs['pred']
        predictions = torch.round(predictions)
        correct_predictions += (predictions == y.to(device)).sum()
        num_predictions += predictions.shape[0]

    accuracy = correct_predictions / num_predictions
    return accuracy

#The training class is similar to that of the first method except that here i am sending 3 inputs to the model
# rather the one i,e each individual vector from dataloader is actually list of four .
#The code is well written and is very much self explanatory
class Trainer_rnn():
    def __init__(self, model, optimizer, device, evaluate_accuracy: Callable = None):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.accuracy = evaluate_accuracy
        self.model.train()
        self.model.to(self.device)

    def train(self,load_data, train_path, dev_path, output_folder,batch=32,epochs=5, interval=5):
        train_dataset = load_data.get_data([json.loads(line) for line in open(train_path, 'r',encoding="utf8")],'train',batch)
        test_dataset =  load_data.get_data([json.loads(line) for line in open(dev_path, 'r',encoding="utf8")],'dev', batch)
        train_history = []
        print(epochs)
        for epoch in range(epochs):
            epoch_loss = 0.0
            len_train = 0

            # each element (sample) in train_dataset is a batch
            for step, sample in tqdm(enumerate(train_dataset), desc="Batch", leave=False):
                inputs = sample[0].to(self.device)
                x_length = sample[1].to(self.device)
                x1_length = sample[2].to(self.device)
                targets = sample[3].to(self.device)
                #                 print(targets)
                batch_out = self.model(inputs, x_length, x1_length, targets)
                loss = batch_out['loss']

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                len_train += 1
            if (epoch % interval) == 0:
                avg_epoch_loss = epoch_loss / len_train
                train_history.append(avg_epoch_loss)
                acc = self.accuracy(self.model, test_dataset,self.device)
                print('Epoch: {} avg loss = {:0.4f} avg acc = {:0.4f}'.format(epoch, avg_epoch_loss, acc))

                torch.save(self.model.state_dict(),
                           os.path.join(output_folder, 'Rnnstate_{}.pt'.format(epoch)))  # save the model state
                # assert self.accuracy is not None

        return {'train_history': train_history}

#same as that of the first method
class StudentModel(Model):
    def __init__(self, model, load_data, device):
        self.model=model
        self.loads = load_data
        self.device =device
        self.model.to(self.device)

    def predict(self, sentence_pairs: List[Dict]) -> List[str]:
        predict_dataloader = self.loads.get_data(sentence_pairs,'predict',len(sentence_pairs))
        for x,z1,z2 in predict_dataloader:
            outputs = self.model(x.to(self.device), z1.to(self.device), z2.to(self.device))
            predictions = outputs['pred']
            predictions = torch.round(predictions)
        return ["True" if x==1 else "False" for x in predictions]


#same as that of the first method
def build_model(device: str) -> Model:
    glove_pretrained_path = 'E:\\Second Semester\\NLP\\homework\\glove.6B.100d.txt'
    train_path= 'E:\\Second Semester\\NLP\\homework\\data\\train.jsonl'
    dev_path= 'E:\\Second Semester\\NLP\\homework\\data\\dev.jsonl'
    output_folder='E:\\Second Semester\\NLP\\homework'
    model_load='E:\\Second Semester\\NLP\\homework\\Rnnstate_55.pt'

    not_trained= 0
    load = RnnDataload(glove_pretrained_path)
    model = SecondNetwork(load.vectors, 120, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.95)
    model.load_state_dict(torch.load(model_load, map_location=device))
    if not_trained:
        trainer = Trainer_rnn(model.float(), optimizer, device,evaluate_accuracy_rnn)
        avg_loss = trainer.train(load,train_path ,dev_path, output_folder,batch=32,epochs=200,interval=5)
    return StudentModel(model, load, device)
md = build_model('cuda')
dev_path= 'E:\\Second Semester\\NLP\\homework\\data\\dev.jsonl'
data =[json.loads(line) for line in open(dev_path, 'r',encoding="utf8")]
md.predict(test)

