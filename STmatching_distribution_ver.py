import numpy as np
import pandas as pd
import networkx as nx

from setting import SetParameter

pd.options.mode.chained_assignment = None
np.seterr(divide='ignore')

# 100m grid
loninter = 0.000976
latinter = 0.0009


def network_data():
    config = SetParameter()
    dataset = str(config.dataset)
    rdnetwork = pd.read_csv(str(config.edge_file), usecols=['section_id', 's_node', 'e_node', 'length'])
    if dataset == 'beijing' or 'porto':
        rdnetwork['length'] = rdnetwork['length'] / 100.0
    roadnetwork = nx.DiGraph()
    for row in rdnetwork.values:
        roadnetwork.add_edge(int(row[1]), int(row[2]), distance=row[-1])
    return roadnetwork
