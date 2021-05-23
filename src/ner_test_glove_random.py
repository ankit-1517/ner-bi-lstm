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


def run_full_code_glove_random(device, model_file, test_data_file, output_file, vocabulary_input_file):
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

	# read vocab file
	def string2dict(s):
		s = s.strip(" \n").split(" ")
		d = {}
		for i in range(0, len(s), 2):
			d[s[i]] = int(s[i+1])
		return d
	
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

	np.random.seed(0)
	with open(vocabulary_input_file) as f:
		l = f.readlines()
	tagtoidx = string2dict(l[0])
	vocabtoidx = string2dict(l[1])
	
	count = len(vocabtoidx)
	pad_idx = count
	count += 1
	unknown_idx = count
	count += 1
	vocab_size = count

	test_data, test_label = prepare_dataset(test_data_file)
	test_dataset = CustomDataset(test_data, test_label)
	test_data_loader = DataLoader(test_dataset, batch_size = 128, shuffle=False, collate_fn = collate_fn_padd)
	criterion = nn.CrossEntropyLoss()

	class lstm_Model(nn.Module):
		def __init__(self, vocab_size, embedding_dim, hidden_dim, number_of_tags):
			super(lstm_Model, self).__init__()
			self.embedding = nn.Embedding(vocab_size, embedding_dim)
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
	
	# LOAD MODEL 
	model = lstm_Model(vocab_size, 100, 100, 17)
	model.load_state_dict(torch.load(model_file))
	model = model.to(device)

	def evaluate_final_test():  
		model.eval()
		total_loss, total_accuracy = 0, 0  
		correct = 0
		total = 0
		Y_test_final = []
		pred_test_final = []
		for step,b in enumerate(test_data_loader): 
			X_test, Y_test = b
			for i in range(Y_test.shape[0]):
				Y_test_final.append(Y_test[i].detach().cpu().numpy().tolist())
			Y_test = Y_test.reshape(Y_test.shape[0]*Y_test.shape[1])
			with  torch.no_grad():
				Yp = model(X_test)
				loss = criterion(Yp, Y_test)
				total_loss = total_loss + loss.item()
				Yp = Yp.argmax(axis = 1)
				Yp = Yp.reshape(Yp.shape[0])
				temp = ((Yp) == Y_test)
				correct += temp.sum().float()
				total += temp.shape[0]
				pred_test_final = np.hstack((pred_test_final, Yp.detach().cpu().numpy()))
		avg_loss = total_loss / len(test_data_loader) 
		accuracy = correct/(total)
		return avg_loss, accuracy, Y_test_final, pred_test_final
	
	avg_loss, accuracy, Y_test_final, pred_test_final = evaluate_final_test()
	
	prediction_test_final = []
	index = 0
	for x in Y_test_final:
		prediction_test_final.append([])
		for y in x:
			prediction_test_final[-1].append(pred_test_final[index])
			index += 1
	idx2tag = {}
	for x in tagtoidx:
		idx2tag[tagtoidx[x]] = x

	def list2labels(l):
		ans = []
		for x in l:
			ans.append([])
			for y in x:
				ans[-1].append(idx2tag[y])
		return ans

	actual = list2labels(Y_test_final)
	pred = list2labels(prediction_test_final)
	print(class_report(actual, pred))

	# get predictions without padding
	def pred2labels(pred, labels):
		l = []
		for i in range(len(labels)):
			l.append([])
			for j in range(len(labels[i])):
				l[-1].append(pred[i][j])
		return l

	def read_full_file(path):
		with open(path) as f:
			l = f.readlines()
		l = [x.strip() for x in l]
		while len(l[0]) == 0:
			l = l[1:]
		full = []
		temp = []
		for i in range(len(l)):
			if len(l[i]) == 0:
				full.append([])
				for x in temp:
					full[-1].append(x.split(" "))
				temp = []
			else:
				temp.append(l[i])
		return full

	full = read_full_file(test_data_file)
	pred = pred2labels(pred, test_label)

	s = "\n"
	for i in range(len(full)):
		for j in range(len(full[i])):
			for k in range(3):
				s += full[i][j][k] + " "
			s += pred[i][j] + "\n"
		s += "\n"

	with open(output_file, "w") as f:
		f.write(s)
	
	
