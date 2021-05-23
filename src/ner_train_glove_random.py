import sys
import contextlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from seqeval.metrics import classification_report as class_report
import copy


def run_full_code_glove_random(device, with_glove_embeddings, model_output_file, data_dir, glove_embeddings_file, vocabulary_output_file):
    path = data_dir
    if path.endswith("/") == False:
        path += "/"

    def load_glove_emb(path_glove):
        word2vectors, word2id = {}, {}
        count = 0
        with open(f'{path_glove}', 'rb') as f:
            for l in f:
                line = l.decode().split()
                word = line[0]
                word2vectors[word] = np.array(line[1:]).astype(np.float)
                word2id[word] = count
                count +=1
        return word2vectors, word2id

    word2vectors, word2id = load_glove_emb(glove_embeddings_file)

    def read_file(path):
        with open(path) as f:
            l = f.readlines()
        l = [x.strip() for x in l]
        while len(l[0]) == 0:
            l = l[1:]
        token = []
        label = []
        temp = []
        for i in range(len(l)):
            if len(l[i]) == 0:
                label.append([])
                token.append([])
                for x in temp:
                    x = x.split(" ")
                    token[-1].append(x[0])
                    label[-1].append(x[3])
                temp = []
            else:
                temp.append(l[i])
        return token, label

    # Using readlines()
    file1 = open(path + 'train.txt', 'r')
    Lines = file1.readlines()
    count = 0
    counttag = 0
    # Strips the newline character
    vocabtoidx = {}
    idxtovocab = {}
    countvocab = {}
    tagtoidx = {}
    idxtotag = {}
    for line in Lines:
        temp = line.strip().split(' ')
        if(len(temp[0]) == 0 ):
            continue
        if((temp[0] not in countvocab)):
            countvocab[temp[0]] = 1
        else:
            countvocab[temp[0]] += 1
    threshold = 2
    for line in Lines:
        temp = line.strip().split(' ')
        if(len(temp[0]) == 0 ):
            continue
        if((temp[0] not in vocabtoidx) and countvocab[temp[0]] >= threshold):
            vocabtoidx[temp[0]] = count
            idxtovocab[count] = temp[0]
            count += 1
        if((temp[3] not in tagtoidx)):
            tagtoidx[temp[3]] = counttag
            idxtotag[counttag] = temp[3] 
            counttag += 1

    pad_idx = count
    count += 1
    unknown_idx = count
    count += 1
    vocab_size = count

    # dump dicts to vocab_file 
    def dict2string(d):
        s = ""
        for x in d:
            s += x + " " + str(d[x]) + " "
        return s

    with open(vocabulary_output_file, "w") as f:
        f.write(dict2string(tagtoidx) + "\n")
        f.write(dict2string(vocabtoidx) + "\n")

    def prepare_dataset(path):
        data, label = read_file(path)
        for i in range(len(data)):
            for j in range(len(data[i])):
                if(data[i][j] in vocabtoidx):
                    data[i][j] = vocabtoidx[data[i][j]]
                else:
                    data[i][j] = unknown_idx
                label[i][j] = tagtoidx[label[i][j]]
        return data , label

    train_data, train_label = prepare_dataset(path + "train.txt")
    val_data, val_label = prepare_dataset(path + "dev.txt")
    # test_data, test_label = prepare_dataset(path + "data/test.txt")

    np.random.seed(0)
    embedding = np.random.uniform(low = -1, high = 1, size=(vocab_size, 100))
    if with_glove_embeddings:
        for i in range(vocab_size-2):
            if(idxtovocab[i] in word2vectors):
                embedding[i] = word2vectors[idxtovocab[i]]

    class CustomDataset(Dataset):
        def __init__(self, data, label):
            self.data = data
            self.label = label
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx], self.label[idx]

    def collate_fn_padd(batch):
        sentences = []
        labels = []
        for x in batch:
            sent, lab = x
            sentences.append(torch.Tensor(sent).to(device))
            labels.append(torch.Tensor(lab).to(device))
        sentences = torch.nn.utils.rnn.pad_sequence(sentences, padding_value = pad_idx, batch_first = True)
        labels = torch.nn.utils.rnn.pad_sequence(labels, padding_value = 0, batch_first = True)
        return sentences.to(torch.int64), labels.to(torch.int64)

    train_dataset = CustomDataset(train_data, train_label)
    val_dataset = CustomDataset(val_data, val_label)
    train_data_loader = DataLoader(train_dataset, batch_size = 128, shuffle=True, collate_fn = collate_fn_padd)
    val_data_loader = DataLoader(val_dataset, batch_size = 128, shuffle=True, collate_fn = collate_fn_padd)

    class lstm_Model(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, number_of_tags, embedding_matrix = None):
            super(lstm_Model, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            if with_glove_embeddings:
                self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix,dtype=torch.float32))
            self.dropout = nn.Dropout(p = 0.5)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional = True)
            self.fc = nn.Linear(2*hidden_dim, number_of_tags)

        def forward(self, x):
            x = x.to(torch.int64)
            x = x.to(device)
            x = self.embedding(x)
            x = self.dropout(x)
            x, _ = self.lstm(x)  
            x = x.contiguous()
            x = x.view(-1, 200)  
            x = self.fc(x)
            return x

    # x = torch.ones((2, 32, 200))
    if with_glove_embeddings:
        model = lstm_Model(vocab_size, 100, 100, 17, embedding)
    else:
        model = lstm_Model(vocab_size, 100, 100, 17)
    model = model.to(device)
    # sample = torch.ones(2, 32)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9, weight_decay = 0.0001)

    train_loss = []
    def train(epoch_number):
        model.train()
        tloss = 0
        for s,b in enumerate(train_data_loader):
            X_train, Y_train = b
            Y_train = Y_train.reshape(Y_train.shape[0]*Y_train.shape[1])
            optimizer.zero_grad()        
            Yp = model(X_train)

            loss = criterion(Yp, Y_train)
            loss.backward()
            tloss = tloss + loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        avgloss = tloss / len(train_data_loader)
        train_loss.append(avgloss)
        return avgloss

    val_loss = []
    def evaluate():  
        model.eval()
        total_loss, total_accuracy = 0, 0  
        correct = 0
        total = 0
        for step,b in enumerate(val_data_loader): 
            X_val, Y_val = b
            Y_val = Y_val.reshape(Y_val.shape[0]*Y_val.shape[1])
            with  torch.no_grad():
                Yp = model(X_val)
                loss = criterion(Yp, Y_val)
                total_loss = total_loss + loss.item()
                Yp = Yp.argmax(axis = 1)
                Yp = Yp.reshape(Yp.shape[0])
                temp = ((Yp) == Y_val)
                correct += temp.sum().float()
                total += temp.shape[0]
        avg_loss = total_loss / len(val_data_loader) 
        accuracy = correct/(total)
        return avg_loss, accuracy
    
    best_val_loss = float('inf')
    nepochs = 100
    for epoch in range(nepochs):
        print('\n Epoch {:} / {:}'.format(epoch + 1, nepochs))
        tloss = train(epoch)
        vloss, valreport = evaluate()
        if vloss < best_val_loss:
            best_val_loss = vloss
            torch.save(model.state_dict(), model_output_file)
        print(f'\nTraining Loss: {tloss:.3f}')
        print(f'Validation Loss: {vloss:.3f}')
        print(f'Validation Report: {valreport}')

    
