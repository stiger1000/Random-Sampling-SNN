import torch.nn as nn
from graph import Node, get_graph_info, get_skip_graph, build_graph, save_graph, load_graph
import torch
import math
import os
from spikingjelly.clock_driven import neuron, surrogate, functional, layer
import torch.nn.functional as F


class depthwise_separable_conv_3x3(nn.Module):
    def __init__(self, nin, nout, stride):
        super(depthwise_separable_conv_3x3, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, stride=stride, padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)
        self.nin = nin
        self.nout = nout
        self.stride = stride

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class SCN(nn.Module):
    def __init__(self, inplanes, outplanes, neuron_type, tau, dropout_rate, every_node_dropout, stride=1):
        super(SCN, self).__init__()
        self.conv = depthwise_separable_conv_3x3(inplanes, outplanes, stride)
        self.bn = nn.BatchNorm2d(outplanes)
        if neuron_type == 'LIF':
            self.sn = neuron.MultiStepLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, v_reset=0., tau=tau, decay_input=False)
        elif neuron_type == 'IF':
            self.sn = neuron.MultiStepIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, v_reset=0.)
        else:
            raise NotImplementedError
        # self.sn = nn.ReLU()
        if every_node_dropout:
            self.dropout = layer.MultiStepDropout(p=dropout_rate)
        else:
            self.dropout = nn.Identity()

    def forward(self, x):
        out = self.sn(x)
        # add(out)
        out = self.dropout(out)
        out = functional.seq_to_ann_forward(out, self.conv)
        out = functional.seq_to_ann_forward(out, self.bn)
        return out


class Node_OP(nn.Module):
    def __init__(self, Node, inplanes, outplanes, neuron_type, tau, dropout_rate, every_node_dropout):
        super(Node_OP, self).__init__()
        self.is_input_node = Node.type == 0
        self.input_nums = len(Node.inputs)
        if self.is_input_node:
            self.conv = SCN(inplanes, outplanes, stride=2, neuron_type=neuron_type, tau=tau,
                                     dropout_rate=dropout_rate, every_node_dropout=every_node_dropout)
        else:
            self.conv = SCN(outplanes, outplanes, stride=1, neuron_type=neuron_type, tau=tau,
                                     dropout_rate=dropout_rate, every_node_dropout=every_node_dropout)

    def forward(self, *input):
        l = len(input)
        if l > 1:
            out = 0
            for i in range(l):
                out += input[i]
        else:
            out = input[0]
        out = self.conv(out)
        return out


class StageBlock(nn.Module):
    def __init__(self, graph, inplanes, outplanes, neuron_type, tau, no_skip, skip_ratio, dropout_rate, every_node_dropout):
        super(StageBlock, self).__init__()
        self.nodes, self.input_nodes, self.output_nodes = get_graph_info(graph)
        self.nodeop = nn.ModuleList()
        for node in self.nodes:
            self.nodeop.append(Node_OP(node, inplanes, outplanes, neuron_type, tau, dropout_rate, every_node_dropout))
        self.no_skip = no_skip
        if not self.no_skip:
            self.skip_graph = get_skip_graph(self.nodes, self.input_nodes, self.output_nodes, skip_ratio)

    def forward(self, x):
        results = {}
        for id in self.input_nodes:
            results[id] = self.nodeop[id](x)
        for id, node in enumerate(self.nodes):
            if id not in self.input_nodes:
                inputs = []
                for _id in node.inputs:
                    inputs.append(results[_id])
                if not self.no_skip:
                    for _id in self.skip_graph[id]:
                        inputs.append(F.pad(results[_id][:-1], pad=(0, 0, 0, 0, 0, 0, 0, 0, 1, 0), mode='constant', value=0))
                results[id] = self.nodeop[id](*inputs)
        result = 0
        for id in self.output_nodes:
            result += results[id]
        result = result / len(self.output_nodes)
        return result


class Tiny(nn.Module):
    def __init__(self, args, num_classes=10):
        super(Tiny, self).__init__()
        print('Building TANet-Tiny!')
        assert(args.dataset == 'CIFAR-10' or args.dataset == 'CIFAR-100')
        self.T = args.T
        self.channels = args.channels
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, args.channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(args.channels)
        if args.neuron_type == 'LIF':
            self.sn1 = neuron.MultiStepLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, v_reset=0.,
                                               tau=args.tau, decay_input=False)
            self.sn2 = neuron.MultiStepLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, v_reset=0.,
                                               tau=args.tau, decay_input=False)
            self.sn = neuron.MultiStepLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, v_reset=0.,
                                              tau=args.tau, decay_input=False)
        elif args.neuron_type == 'IF':
            self.sn1 = neuron.MultiStepIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, v_reset=0.)
            self.sn2 = neuron.MultiStepIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, v_reset=0.)
            self.sn = neuron.MultiStepIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, v_reset=0.)
        else:
            raise NotImplementedError
        self.conv2 = nn.Conv2d(args.channels, args.channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(args.channels)
        if args.resume:
            graph = load_graph(os.path.join(args.save_path, 'conv3.yaml'))
        else:
            graph = build_graph(args.nodes, args)
            save_graph(graph, os.path.join(args.save_path, 'conv3.yaml'))
        print(graph)
        self.conv3 = StageBlock(graph, args.channels, args.channels * 2, args.neuron_type, args.tau, args.no_skip,
                                args.skip_ratio, args.dropout_rate, args.every_node_dropout)
        if args.resume:
            graph = load_graph(os.path.join(args.save_path, 'conv4.yaml'))
        else:
            graph = build_graph(args.nodes, args)
            save_graph(graph, os.path.join(args.save_path, 'conv4.yaml'))
        self.conv4 = StageBlock(graph, args.channels * 2, args.channels * 4, args.neuron_type, args.tau,
                                args.no_skip, args.skip_ratio, args.dropout_rate, args.every_node_dropout)
        self.conv = nn.Conv2d(args.channels * 4, 1280, kernel_size=1)
        self.bn = nn.BatchNorm2d(1280)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = layer.MultiStepDropout(p=args.dropout_rate)
        self.fc = nn.Linear(1280, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        x = self.sn1(x)
        
        x = functional.seq_to_ann_forward(x, self.conv2)
        x = functional.seq_to_ann_forward(x, self.bn2)
        # x = self.sn2(x)
        
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.sn2(x)

        x = functional.seq_to_ann_forward(x, self.conv)
        x = functional.seq_to_ann_forward(x, self.bn)
        x = self.sn(x)

        x = functional.seq_to_ann_forward(x, self.avgpool)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.dropout(x)
        x = functional.seq_to_ann_forward(x, self.fc)
        return x


class Regular(nn.Module):
    def __init__(self, args, num_classes=1000):
        super(Regular, self).__init__()
        print('Building TANet-Regular!')
        assert(args.dataset == 'IMAGENET')
        self.T = args.T
        self.channels = args.channels
        self.num_classes = num_classes

        self.conv1 = depthwise_separable_conv_3x3(3, args.channels // 2, 2)
        self.bn1 = nn.BatchNorm2d(args.channels // 2)
        if args.neuron_type == 'LIF':
            self.sn1 = neuron.MultiStepLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, v_reset=0.,
                                               tau=args.tau, decay_input=False)
            self.sn2 = neuron.MultiStepLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, v_reset=0.,
                                               tau=args.tau, decay_input=False)
        elif args.neuron_type == 'IF':
            self.sn1 = neuron.MultiStepIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, v_reset=0.)
            self.sn2 = neuron.MultiStepIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, v_reset=0.)
        else:
            raise NotImplementedError
        if args.resume:
            graph = load_graph(os.path.join(args.save_path, 'conv2.yaml'))
        else:
            graph = build_graph(args.nodes // 2, args)
            save_graph(graph, os.path.join(args.save_path, 'conv2.yaml'))
        self.conv2 = StageBlock(graph, args.channels // 2, args.channels, args.neuron_type, args.tau,
                                args.no_skip, args.skip_ratio, args.dropout_rate, args.every_node_dropout)
        if args.resume:
            graph = load_graph(os.path.join(args.save_path, 'conv3.yaml'))
        else:
            graph = build_graph(args.nodes, args)
            save_graph(graph, os.path.join(args.save_path, 'conv3.yaml'))
        self.conv3 = StageBlock(graph, args.channels, args.channels * 2, args.neuron_type, args.tau,
                                args.no_skip, args.skip_ratio, args.dropout_rate, args.every_node_dropout)
        if args.resume:
            graph = load_graph(os.path.join(args.save_path, 'conv4.yaml'))
        else:
            graph = build_graph(args.nodes, args)
            save_graph(graph, os.path.join(args.save_path, 'conv4.yaml'))
        self.conv4 = StageBlock(graph, args.channels * 2, args.channels * 4, args.neuron_type, args.tau,
                                args.no_skip, args.skip_ratio, args.dropout_rate, args.every_node_dropout)
        if args.resume:
            graph = load_graph(os.path.join(args.save_path, 'conv5.yaml'))
        else:
            graph = build_graph(args.nodes, args)
            save_graph(graph, os.path.join(args.save_path, 'conv5.yaml'))
        self.conv5 = StageBlock(graph, args.channels * 4, args.channels * 8, args.neuron_type, args.tau,
                                args.no_skip, args.skip_ratio, args.dropout_rate, args.every_node_dropout)
        # self.relu = nn.ReLU()
        self.conv = nn.Conv2d(args.channels * 8, 1280, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(1280)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(1280, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        # x = self.sn1(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.sn1(x)

        x = functional.seq_to_ann_forward(x, self.conv)
        x = functional.seq_to_ann_forward(x, self.bn2)
        x = self.sn2(x)

        x = functional.seq_to_ann_forward(x, self.avgpool)
        x = x.view(x.size(0), x.size(1), -1)
        x = functional.seq_to_ann_forward(x, self.fc)

        return x
