from client.client_torch import FedAvgClientTorch
from client.client_torch.fedper_client_torch import FedPerClientTorch
from ..fedpredict_core import fedpredict_core
from torch.nn.parameter import Parameter
import torch
import json
import math
from pathlib import Path
import numpy as np
import flwr
import json
from utils.quantization.fetsgd import layers_sketching
import os
import sys
import time
import pandas as pd

import warnings
warnings.simplefilter("ignore")

import logging
# logging.getLogger("torch").setLevel(logging.ERROR)

class FetSGDClientTorch(FedAvgClientTorch):

	def __init__(self,
				 cid,
				 n_clients,
				 n_classes,
				 args,
				 epochs=1,
				 model_name         = 'DNN',
				 client_selection   = False,
				 strategy_name      ='FedPredict',
				 aggregation_method = 'None',
				 dataset            = '',
				 perc_of_clients    = 0,
				 decay              = 0,
				 fraction_fit		= 0,
				 non_iid            = False,
				 m_combining_layers	= 1,
				 new_clients			= False,
				 new_clients_train	= False
				 ):

		super().__init__(cid=cid,
						 n_clients=n_clients,
						 n_classes=n_classes,
						 args=args,
						 epochs=epochs,
						 model_name=model_name,
						 client_selection=client_selection,
						 strategy_name=strategy_name,
						 aggregation_method=aggregation_method,
						 dataset=dataset,
						 perc_of_clients=perc_of_clients,
						 decay=decay,
						 fraction_fit=fraction_fit,
						 non_iid=non_iid,
						 new_clients=new_clients,
						 new_clients_train=new_clients_train)

		self.m_combining_layers = [i for i in range(len([i for i in self.create_model().parameters()]))]
		self.lr_loss = torch.nn.MSELoss()
		self.clone_model = self.create_model().to(self.device)
		self.round_of_last_fit = 0
		self.rounds_of_fit = 0
		self.T = int(args.T)
		self.accuracy_of_last_round_of_fit = 0
		self.start_server = 0
		self.filename = """./{}_saved_weights/{}/{}/model.pth""".format(strategy_name.lower(), self.model_name, self.cid)

	def save_parameters(self):
		# Using 'torch.save'
		try:
			# filename = """./fedpredict_saved_weights/{}/{}/model.pth""".format(self.model_name, self.cid)
			if Path(self.filename).exists():
				os.remove(self.filename)
			torch.save(self.model.state_dict(), self.filename)
		except Exception as e:
			print("save parameters")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def _fedpredict_plugin(self, global_parameters, t, T, nt, M):

		try:

			local_model_weights, global_model_weight = fedpredict_core(t, T, nt)

			# Load global parameters into 'self.clone_model' (global model)
			global_parameters = [Parameter(torch.Tensor(i.tolist())) for i in global_parameters]
			local_layer_count = 0
			global_layer_count = 0
			parameters = [Parameter(torch.Tensor(i.tolist())) for i in global_parameters]

			for old_param in self.clone_model.parameters():
				if local_layer_count in M:
					new_param = parameters[global_layer_count]
					old_param.data = new_param.data.clone()
					global_layer_count += 1
				local_layer_count += 1

			# self.clone_model.load_state_dict(torch.load(filename))
			# Combine models
			count = 0
			for new_param, old_param in zip(self.clone_model.parameters(), self.model.parameters()):
				if count in self.m_combining_layers:
					old_param.data = (global_model_weight*new_param.data.clone() + local_model_weights*old_param.data.clone())
				count += 1

		except Exception as e:
			print("merge models")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def set_parameters_to_model_evaluate(self, global_parameters, config={}):
		# Using 'torch.load'
		try:
			print("Dimensões: ", [i.detach().numpy().shape for i in self.model.parameters()])
			print("Dimensões recebidas: ", [i.shape for i in global_parameters])
			global_parameters = layers_sketching(global_parameters, model_shape=[i.detach().numpy().shape for i in self.model.parameters()])
			parameters = [Parameter(torch.Tensor(i.tolist())) for i in global_parameters]
			for new_param, old_param in zip(parameters, self.model.parameters()):
				old_param.data = new_param.data.clone()
		except Exception as e:
			print("Set parameters to model")
			print('Error on line {} client id {}'.format(sys.exc_info()[-1].tb_lineno, self.cid), type(e).__name__, e)