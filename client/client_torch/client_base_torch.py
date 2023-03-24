import flwr as fl
import tensorflow
import random
import time
import numpy as np
import torch
import os
import time
import sys

from dataset_utils_torch import ManageDatasets
from model_definition_torch import DNN, Logistic, CNN
import csv
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader
import warnings
warnings.simplefilter("ignore")

import logging
# logging.getLogger("torch").setLevel(logging.ERROR)
from torch.nn.parameter import Parameter
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
class ClientBaseTorch(fl.client.NumPyClient):

	def __init__(self,
				 cid,
				 n_clients,
				 n_classes,
				 epochs				= 1,
				 model_name         = 'None',
				 client_selection   = False,
				 solution_name      = 'None',
				 aggregation_method = 'None',
				 dataset            = '',
				 perc_of_clients    = 0,
				 decay              = 0,
				 non_iid            = False,
				 new_clients = False,
				 new_clients_train	= False
				 ):

		self.cid          = int(cid)
		self.n_clients    = n_clients
		self.model_name   = model_name
		self.local_epochs = epochs
		self.non_iid      = non_iid

		self.num_classes = n_classes

		self.model        = None
		self.x_train      = None
		self.x_test       = None
		self.y_train      = None
		self.y_test       = None

		#logs
		self.solution_name      = solution_name
		self.aggregation_method = aggregation_method
		self.dataset            = dataset

		self.client_selection = client_selection
		self.perc_of_clients  = perc_of_clients
		self.decay            = decay

		self.loss = nn.CrossEntropyLoss()
		self.learning_rate = 0.01
		self.new_clients = new_clients
		self.new_clients_train = new_clients_train
		# self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.type = 'torch'

		#params
		if self.aggregation_method == 'POC':
			self.solution_name = f"{solution_name}-{aggregation_method}-{self.perc_of_clients}"

		elif self.aggregation_method == 'FedLTA': 
			self.solution_name = f"{solution_name}-{aggregation_method}-{self.decay}"

		elif self.aggregation_method == 'None':
			self.solution_name = f"{solution_name}-{aggregation_method}"

		self.base = f"logs/{self.type}/{self.solution_name}/new_clients_{self.new_clients}_train_{self.new_clients_train}/{self.n_clients}/{self.model_name}/{self.dataset}/{self.local_epochs}_local_epochs"
		self.evaluate_client_filename = f"{self.base}/evaluate_client.csv"
		self.train_client_filename = f"{self.base}/train_client.csv"

		self.trainloader, self.testloader = self.load_data(self.dataset, n_clients=self.n_clients)
		self.model                                           = self.create_model().to(self.device)
		# self.device = 'cpu'
		self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

	def load_data(self, dataset_name, n_clients, batch_size=32):
		try:
			x_train, y_train, x_test, y_test = ManageDatasets(self.cid, self.model_name).select_dataset(dataset_name, n_clients, self.non_iid)
			# print("y test")
			# print(np.unique(y_test))
			# exit()
			self.input_shape = x_train.shape
			tensor_x_train = torch.Tensor(x_train)  # transform to torch tensor
			tensor_y_train = torch.Tensor(y_train)

			train_dataset = TensorDataset(tensor_x_train, tensor_y_train)
			trainLoader = DataLoader(train_dataset, batch_size, drop_last=True, shuffle=True)

			tensor_x_test = torch.Tensor(x_test)  # transform to torch tensor
			tensor_y_test = torch.Tensor(y_test)

			test_dataset = TensorDataset(tensor_x_test, tensor_y_test)
			testLoader = DataLoader(test_dataset, batch_size, drop_last=True, shuffle=True)

			return trainLoader, testLoader
		except Exception as e:
			print("load data")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def create_model(self):

		try:
			# print("tamanho: ", self.input_shape, " dispositivo: ", self.device)
			if self.dataset in ['MNIST', 'CIFAR10']:
				input_shape = self.input_shape[1]*self.input_shape[2]
			if self.model_name == 'Logist Regression':
				return Logistic(input_shape, self.num_classes)
			elif self.model_name == 'DNN':
				if self.dataset == 'UCIHAR':
					input_shape = self.input_shape[1]
				return DNN(input_shape=input_shape, num_classes=self.num_classes)
			elif self.model_name == 'CNN':
				if self.dataset == 'MNIST':
					input_shape = 1
					mid_dim = 256
				else:
					input_shape = 3
					mid_dim = 400
				return CNN(input_shape=input_shape, num_classes=self.num_classes, mid_dim=mid_dim)
			else:
				raise Exception("Wrong model name")
		except Exception as e:
			print("create model")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


	def get_parameters(self, config):
		try:
			parameters = [i.detach().cpu().numpy() for i in self.model.parameters()]
			return parameters
		except Exception as e:
			print("get parameters")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

		# It does the same of "get_parameters", but using "get_parameters" in outside of the core of Flower is causing errors
	def get_parameters_of_model(self):
		try:
			parameters = [i.detach().numpy() for i in self.model.parameters()]
			return parameters
		except Exception as e:
			print("get parameters of model")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def set_parameters(self, parameters):
		try:
			parameters = [Parameter(torch.Tensor(i.tolist())) for i in parameters]
			for new_param, old_param in zip(parameters, self.model.parameters()):
				old_param.data = new_param.data.clone()
		except Exception as e:
			print("set parameters")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def clone_model(self, model, target):
		try:
			for param, target_param in zip(model.parameters(), target.parameters()):
				target_param.data = param.data.clone()
			# target_param.grad = param.grad.clone()
		except Exception as e:
			print("clone model")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def update_parameters(self, model, new_params):
		for param, new_param in zip(model.parameters(), new_params):
			param.data = new_param.data.clone()

	def set_parameters_to_model(self, parameters):
		try:
			parameters = [Parameter(torch.Tensor(i.tolist())) for i in parameters]
			for new_param, old_param in zip(parameters, self.model.parameters()):
				old_param.data = new_param.data.clone()
		except Exception as e:
			print("set parameters to model")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def save_parameters(self):
		pass

	def fit(self, parameters, config):
		try:
			selected_clients   = []
			trained_parameters = []
			selected           = 0

			if config['selected_clients'] != '':
				selected_clients = [int (cid_selected) for cid_selected in config['selected_clients'].split(' ')]

			start_time = time.process_time()
			#print(config)
			if self.cid in selected_clients or self.client_selection == False or int(config['round']) == 1:
				self.set_parameters_to_model(parameters)

				selected = 1
				self.model.train()

				start_time = time.time()

				max_local_steps = self.local_epochs
				train_acc = 0
				train_loss = 0
				train_num = 0
				for step in range(max_local_steps):
					for i, (x, y) in enumerate(self.trainloader):
						if type(x) == type([]):
							x[0] = x[0].to(self.device)
						else:
							x = x.to(self.device)
						y = y.to(self.device)
						train_num += y.shape[0]

						self.optimizer.zero_grad()
						output = self.model(x)
						y = torch.tensor(y.int().detach().numpy().astype(int).tolist())
						loss = self.loss(output, y)
						train_loss += loss.item() * y.shape[0]
						loss.backward()
						self.optimizer.step()

						train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()

				trained_parameters = self.get_parameters_of_model()
				self.save_parameters()

			total_time         = time.process_time() - start_time
			size_of_parameters = sum([sum(map(sys.getsizeof, trained_parameters[i])) for i in range(len(trained_parameters))])
			avg_loss_train     = train_loss/train_num
			avg_acc_train      = train_acc/train_num

			data = [config['round'], self.cid, selected, total_time, size_of_parameters, avg_loss_train, avg_acc_train]

			self._write_output(
				filename=self.train_client_filename,
				data=data)

			fit_response = {
				'cid' : self.cid
			}

			return trained_parameters, train_num, fit_response
		except Exception as e:
			print("fit")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


	def evaluate(self, parameters, config):
		try:
			self.set_parameters_to_model(parameters)
			# loss, accuracy     = self.model.evaluate(self.x_test, self.y_test, verbose=0)
			self.model.eval()

			test_acc = 0
			test_loss = 0
			test_num = 0

			with torch.no_grad():
				for x, y in self.testloader:
					if type(x) == type([]):
						x[0] = x[0].to(self.device)
					else:
						x = x.to(self.device)
					self.optimizer.zero_grad()
					y = y.to(self.device)
					y = torch.tensor(y.int().detach().numpy().astype(int).tolist())
					output = self.model(x)
					loss = self.loss(output, y)
					test_loss += loss.item() * y.shape[0]
					test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
					test_num += y.shape[0]

			size_of_parameters = sum([sum(map(sys.getsizeof, parameters[i])) for i in range(len(parameters))])
			loss = test_loss/test_num
			accuracy = test_acc/test_num
			data = [config['round'], self.cid, size_of_parameters, loss, accuracy]

			self._write_output(filename=self.evaluate_client_filename,
							   data=data)

			evaluation_response = {
				"cid"      : self.cid,
				"accuracy" : float(accuracy)
			}

			return loss, test_num, evaluation_response
		except Exception as e:
			print("evaluate")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def _write_output(self, filename, data):

		with open(filename, 'a') as server_log_file:
			writer = csv.writer(server_log_file)
			writer.writerow(data)

