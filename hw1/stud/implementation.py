# import the already built in libraries
import numpy as np
from typing import List, Tuple, Dict
from tqdm.notebook import tqdm
from typing import *
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import os
import json
import re
import distutils
from model import Model

#in this function all the necessary things important  for the model to run are being wrapped inside the model object.
#loading the already trained model, initializing the dataload function which initializes the constructor for loading.
#the already pretrained glove vectors , here not_trained is turned off "0" so that it will not go into the training loop.
#if we want to train it just turn on the not_trained to 1.
#map_location is just to map weights to correct device while loading.
#rest off the code is very much self explanatory.
def build_model(device: str) -> Model:
    glove_pretrained_path = './model/glove.6B.50d.txt'
    train_path= './data/train.jsonl'
    dev_path= './data/dev.jsonl'
    output_folder='./model'
    model_load='./model/trainedModel.pt'
    device = device
    not_trained= 0
    load_data = Dataload(glove_pretrained_path)
    model = Classifier(101, 120,10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.95)
    model.load_state_dict(torch.load(model_load, map_location=device))
    if not_trained:
        trainer = Trainer(model.float(), optimizer, device,evaluate_accuracy)
        trainer.train(load_data,train_path ,dev_path, output_folder,batch=32,epochs=200,interval=5)
    return StudentModel(model, load_data, device)

#This class have four functions and these are implemented to load the data and convert that into vectors.
#loadPreTrainedWeights is just to load the glove weights at the time of object creation.
#form_vector is to apply the Hierarchical pooling on the vector set to return the single vector.
#tokenize is the function taking in the sentence and tokenizing into the words and then replacing each word with
#embedding from glove vectors.
#get_data is the function which accepts the json object and convert it into the vectors by calling the above functions
#then return the dataloader object having all the vectors and batched into the batch size.
class Dataload():

    def __init__(self, path_pretrained_weights: str):
        super().__init__()
        self.wordVector = self.loadPreTrainedWeights(path_pretrained_weights)

    def loadPreTrainedWeights(self, path_pretrained_weights):
        f = open(path_pretrained_weights, 'r', encoding="utf8")
        wordVector = {}
        for line in f:
            splitLines = line.split()
            word = splitLines[0]
            wordEmbedding = np.array([float(value) for value in splitLines[1:]])
            wordVector[word] = wordEmbedding
        return wordVector

    def form_vector(self, sentence: str) -> Optional[torch.Tensor]:
        check = sentence.shape[0]
        sentence = sentence.view(1, 1, sentence.shape[0], sentence.shape[1])
        # can make global variable
        average_pooling = nn.AvgPool2d((check, 1), stride=(1, 1))
        sentence = average_pooling(sentence)
        max_pooling = nn.MaxPool2d((sentence.shape[2], 1), stride=(sentence.shape[2], 1))
        sentence = max_pooling(sentence)
        sentence = sentence.view(sentence.shape[3])
        return sentence

    def tokenize_line(self, line, pattern='\W', size=50):
        zeros = np.zeros(size)
        vector = [self.wordVector[w] if w in self.wordVector else zeros for w in
                  [word for word in re.split(pattern, line.lower()) if word]]
        if len(vector) == 0:
            return None
        return torch.tensor(vector)

    def get_data(self, data, operation, batch=None):
        sample_data = []
        seperator = torch.tensor([0])
        for parse in data:
            sentence1 = parse['sentence1']
            sentence2 = parse['sentence2']
            word1 = sentence1[int(parse['start1']):int(parse['end1'])]
            word2 = sentence2[int(parse['start2']):int(parse['end2'])]
            if (word1.lower()) != (word2.lower()):
                sentence2 = "".join((sentence2[:int(parse['start2'])], word1, sentence2[int(parse['end2']):]))
            sentence1 = self.form_vector(self.tokenize_line(sentence1))
            sentence2 = self.form_vector(self.tokenize_line(sentence2))
            if None not in [sentence1, sentence2]:
                if (operation == 'train' or operation == 'dev'):
                    label = (1 if distutils.util.strtobool(parse['label']) else 0)
                    sample_data.append((torch.cat((sentence1, seperator, sentence2), 0), torch.tensor(label)))
                else:
                    sample_data.append((torch.cat((sentence1, seperator, sentence2), 0)))
        return DataLoader(sample_data, batch_size=batch)

#This is the function just to evaluate the accuracy for the validation dataset
#it accepts the  validation data loader and returns the accuracy over that
@torch.no_grad()
def evaluate_accuracy(model: nn.Module, dataloader: DataLoader ,device):
    correct_predictions = 0
    num_predictions = 0
    for x, y in dataloader:
        outputs = model(x.to(device).float())
        predictions = outputs['pred']
        predictions = torch.round(predictions)
        correct_predictions += (predictions == y.to(device)).sum()
        num_predictions += predictions.shape[0]

    accuracy = correct_predictions / num_predictions
    return accuracy

#This is main model Architecture class where i have designed simple neural network with the three layers.
#it is extending (inheriting) the nn.module class of torch
#used the relu for non linearity and sigmoid for to get one output and binary cross entropy as loss function
class Classifier(torch.nn.Module):

    def __init__(self, n_features: int, n_hidden: int, n_hidden2: int):
        super().__init__()
        self.lin1 = torch.nn.Linear(n_features, n_hidden)
        self.lin2 = torch.nn.Linear(n_hidden, n_hidden2)
        self.lin3 =  torch.nn.Linear(n_hidden2, 1)
        self.loss_fn = torch.nn.BCELoss()

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
#         print(x,y)
        # actual forward
        out = self.lin1(x)
        out = torch.relu(out)
        out = self.lin2(out)
        out = torch.relu(out)
        out = self.lin3(out).squeeze(1)
        # we need to apply a sigmoid activation function
        out = torch.sigmoid(out)

        result = {'pred': out}

        # compute loss
        if y is not None:
            loss = self.loss(out, y)
            result['loss'] = loss

        return result

    def loss(self, pred, y):
        return self.loss_fn(pred, y)
#This is training class which accepts in constructor the model object and optimizer object which in this case
# Stochastic gradient descent , adding the output folder location to store the model, number of the epochs etc
#I think most of the code is written in well maintained manner and is self explanatory
class Trainer():
    def __init__(self, model, optimizer, device, evaluate_accuracy: Callable = None):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.accuracy = evaluate_accuracy
        self.model.train()
        self.model.to(self.device)

    def train(self,load_data,train_path ,dev_path, output_folder,batch=32,epochs=5,interval=5):
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
                targets = sample[1].to(self.device)
                batch_out = self.model(inputs.float(), targets.float())
                loss = batch_out['loss']

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                len_train += 1
            if (epoch%interval) == 0:
                avg_epoch_loss = epoch_loss / len_train
                train_history.append(avg_epoch_loss)
                acc = self.accuracy(self.model, test_dataset,self.device)
                print('Epoch: {} avg loss = {:0.4f} avg acc = {:0.4f}'.format(epoch, avg_epoch_loss, acc))

                torch.save(self.model.state_dict(),
                           os.path.join(output_folder, 'state_{}.pt'.format(epoch)))  # save the model state
                # assert self.accuracy is not None


        return {'train_history': train_history}



class RandomBaseline(Model):

    options = [
        ('True', 40000),
        ('False', 40000),
    ]

    def __init__(self):

        self._options = [option[0] for option in self.options]
        self._weights = np.array([option[1] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, sentence_pairs: List[Dict]) -> List[str]:
        return [str(np.random.choice(self._options, 1, p=self._weights)[0]) for x in sentence_pairs]

#this is the student class and inherits the Model class from model
#just wrapper for the model , data_loader ,device and have the implementation of the predict function of model class
class StudentModel(Model):
    def __init__(self, model, load_data, device):
        self.model=model
        self.loads = load_data
        self.device =device
        self.model.to(self.device)
    def predict(self, sentence_pairs: List[Dict]) -> List[str]:
        predict_dataloader = self.loads.get_data(sentence_pairs,'predict',len(sentence_pairs))
        for x in predict_dataloader:
            outputs = self.model(x.to(self.device).float())
            predictions = outputs['pred']
            predictions = torch.round(predictions)
        return ["True" if x==1 else "False" for x in predictions]
