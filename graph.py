import networkx as nx
import yaml
import collections
import numpy
import random


Node = collections.namedtuple('Node', ['id', 'inputs', 'type'])


def get_graph_info(graph):
    input_nodes = []
    output_nodes = []
    Nodes = []
    for node in range(graph.number_of_nodes()):
        tmp = list(graph.neighbors(node))
        tmp.sort()
        type = -1
        if len(tmp) == 0:
            input_nodes.append(node)
            output_nodes.append(node)
            type = 0
        else:
            if node < tmp[0]:
                input_nodes.append(node)
                type = 0
            if node > tmp[-1]:
                output_nodes.append(node)
                type = 1
        Nodes.append(Node(node, [n for n in tmp if n < node], type))
    return Nodes, input_nodes, output_nodes


# randomly replace edge based on graph
def get_skip_graph(nodes, input_nodes, output_nodes, skip_ratio):
    skip_graph = []
    for id, node in enumerate(nodes):
        input_id = []
        for _id in node.inputs:
            if random.random() <= skip_ratio and len(input_id) < len(node.inputs) - 1:
                input_id.append(_id)
                # print(_id, id)
        skip_graph.append(input_id)
        for _id in input_id:
            node.inputs.remove(_id)
    return skip_graph


def build_graph(Nodes, args):
    args.graph_seed += 1
    if args.graph_model == 'ER':
        return nx.random_graphs.erdos_renyi_graph(Nodes, args.P, args.graph_seed)
    elif args.graph_model == 'BA':
        return nx.random_graphs.barabasi_albert_graph(Nodes, args.M, args.graph_seed)
    elif args.graph_model == 'WS':
        return nx.random_graphs.connected_watts_strogatz_graph(Nodes, args.K, args.P, tries=200, seed=args.graph_seed)
    elif args.graph_model == 'GNM':
        return nx.random_graphs.gnm_random_graph(Nodes, args.M)

def save_graph(graph, path):
    with open(path, 'w') as f:
        yaml.dump(graph, f)


def load_graph(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.Loader)


def calc_path(graph):
    nodes, input_nodes, output_nodes = get_graph_info(graph)
    num_path = {}
    len_path = {}
    num = 0
    len = 0
    for id, node in enumerate(nodes):
        if id in input_nodes:
            num_path[id] = 1
            len_path[id] = 1
        else:
            num_path[id] = 0
            len_path[id] = 0
            for _id in node.inputs:
                print(_id, id)
                num_path[id] += num_path[_id]
                len_path[id] += len_path[_id] + num_path[_id]
        print(id, num_path[id], len_path[id])
        if id in output_nodes:
            num += num_path[id]
            len += num_path[id] + len_path[id]

    return num, len