import flwr as fl
import numpy as np
import math
import os
import time

from flwr.common import FitIns
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

from server.server_base import ServerBase
from pathlib import Path
import shutil

class FedPerServer(ServerBase):

    def __init__(self, aggregation_method, n_classes, fraction_fit, num_clients,
                 decay=0, perc_of_clients=0, dataset='', strategy_name='FedPer', model_name=''):

        super().__init__(aggregation_method=aggregation_method,
                         n_classes=n_classes,
                         fraction_fit=fraction_fit,
                         num_clients=num_clients,
                         decay=decay,
                         perc_of_clients=perc_of_clients,
                         dataset=dataset,
                         strategy_name='FedPer',
                         model_name=model_name)

        directory = """fedper_saved_weights/{}/""".format(self.model_name)
        if Path(directory).exists():
            shutil.rmtree(directory)