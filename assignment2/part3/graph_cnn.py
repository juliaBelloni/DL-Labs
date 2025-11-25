import torch.nn as nn
import torch

import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, to_dense_adj



class MatrixGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(MatrixGraphConvolution, self).__init__()
        self.W = nn.Parameter(torch.Tensor(out_features, in_features))
        self.B = nn.Parameter(torch.Tensor(out_features, in_features))

        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.B)

    def make_adjacency_matrix(self, edge_index, num_nodes):
        """
        Creates adjacency matrix from edge index.

        :param edge_index: [source, destination] pairs defining directed edges nodes. dims: [2, num_edges]
        :param num_nodes: number of nodes in the graph.
        :return: adjacency matrix with shape [num_nodes, num_nodes]

        Hint: A[i,j] -> there is an edge from node j to node i
        """
        # inefficient solution
        # adjacency_matrix = torch.zeros((num_nodes, num_nodes), device=edge_index.device)
        # for src, dst in edge_index.t():
        #     adjacency_matrix[dst, src] = 1.0

        # more using torch_geometric library
        adjacency_matrix = to_dense_adj(edge_index, max_num_nodes=num_nodes)
        adjacency_matrix = adjacency_matrix.squeeze(0).t().to(dtype=torch.float)
        return adjacency_matrix

    def make_inverted_degree_matrix(self, edge_index, num_nodes):
        """
        Creates inverted degree matrix from edge index.

        :param edge_index: [source, destination] pairs defining directed edges nodes. shape: [2, num_edges]
        :param num_nodes: number of nodes in the graph.
        :return: inverted degree matrix with shape [num_nodes, num_nodes]. Set degree of nodes without an edge to 1.
        """
        degree_vector = torch.bincount(edge_index[1], minlength=num_nodes).to(device=edge_index.device)
        inverted_degree_vector = torch.where(degree_vector==0, 0, 1.0 / degree_vector)
        inverted_degree_matrix = torch.diag(inverted_degree_vector)
        return inverted_degree_matrix

    def forward(self, x, edge_index):
        """
        Forward propagation for GCNs using efficient matrix multiplication.

        :param x: values of nodes. shape: [num_nodes, num_features]
        :param edge_index: [source, destination] pairs defining directed edges nodes. shape: [2, num_edges]
        :return: activations for the GCN
        """
        A = self.make_adjacency_matrix(edge_index, x.size(0))
        D_inv = self.make_inverted_degree_matrix(edge_index, x.size(0))
        
        out = D_inv @ A @ x @ self.W.T + x @ self.B.T
        return out

class MessageGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(MessageGraphConvolution, self).__init__()
        self.W = nn.Parameter(torch.Tensor(out_features, in_features))
        self.B = nn.Parameter(torch.Tensor(out_features, in_features))

        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.B)

    @staticmethod
    def message(x, edge_index):
        """
        message step of the message passing algorithm for GCNs.

        :param x: values of nodes. shape: [num_nodes, num_features]
        :param edge_index: [source, destination] pairs defining directed edges nodes. shape: [2, num_edges]
        :return: message vector with shape [num_nodes, num_in_features]. Messages correspond to the old node values.

        Hint: check out torch.Tensor.index_add function
        """
        messages = x[edge_index[0]]
        aggregated_messages = torch.zeros_like(x, device=x.device)
        aggregated_messages = aggregated_messages.index_add(0, edge_index[1], messages)
        sum_weight = torch.bincount(edge_index[1], minlength=x.size(0)).to(x.dtype).unsqueeze(1)  # [num_nodes, 1]
        sum_weight = sum_weight.clamp_min(1.0)
        aggregated_messages = aggregated_messages / sum_weight
        return aggregated_messages

    def update(self, x, messages):
        """
        update step of the message passing algorithm for GCNs.

        :param x: values of nodes. shape: [num_nodes, num_features]
        :param messages: messages vector with shape [num_nodes, num_in_features]
        :return: updated values of nodes. shape: [num_nodes, num_out_features]
        """
        x = messages @ self.W.T + x @ self.B.T

        return x

    def forward(self, x, edge_index):
        message = self.message(x, edge_index)
        x = self.update(x, message)
        return x


class GraphAttention(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphAttention, self).__init__()
        self.W = nn.Parameter(torch.Tensor(out_features, in_features))
        self.a = nn.Parameter(torch.Tensor(out_features * 2))

        nn.init.xavier_uniform_(self.W)
        nn.init.uniform_(self.a, 0, 1)

    def forward(self, x, edge_index, debug=False):
        """
        Forward propagation for GATs.
        Follow the implementation of Graph attention networks (Veličković et al. 2018).

        :param x: values of nodes. shape: [num_nodes, num_features]
        :param edge_index: [source, destination] pairs defining directed edges nodes. shape: [2, num_edges]
        :param debug: used for tests
        :return: updated values of nodes. shape: [num_nodes, num_out_features]
        :return: debug data for tests:
                 messages -> messages vector with shape [num_nodes + num_edges, num_out_features], i.e. Wh from Veličković et al.
                 edge_weights_numerator -> unnormalized edge weightsm i.e. exp(e_ij) from Veličković et al.
                 softmax_denominator -> per destination softmax normalizer

        Hint: the GAT implementation uses only 1 parameter vector and edge index with self loops
        Hint: It is easier to use/calculate only the numerator of the softmax
              and weight with the denominator at the end.

        Hint: check out torch.Tensor.index_add function
        """
        edge_index, _ = add_self_loops(edge_index)

        sources, destinations = edge_index
        
        
        activations = x @ self.W.T # [num_nodes, num_out_features]
        messages = torch.cat([activations[destinations], activations[sources]], dim = -1) # [num_edges, 2*num_out_features]
        print(x.shape, activations.shape)
        print(x[sources].shape, x[destinations].shape)

        attention_inputs = F.leaky_relu(messages @ self.a) # [num_edges, 1]

        edge_weights_numerator = torch.exp(attention_inputs) # [num_edges, 1]

        weighted_messages =  activations[sources] * edge_weights_numerator.reshape(-1, 1)  # [num_edges, num_out_features]

        softmax_denominator = torch.zeros(x.size(0), device=x.device).index_add(0, destinations, edge_weights_numerator)  # [num_nodes]

        aggregated_messages = torch.zeros_like(activations, device=x.device).index_add(0, destinations, weighted_messages)  / softmax_denominator.reshape(-1, 1)  # [num_nodes, num_out_features]
        
        if debug:
            return aggregated_messages, {'edge_weights': edge_weights_numerator, 'softmax_weights': softmax_denominator,
                                        'messages': messages}
        return aggregated_messages
