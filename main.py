import math, random


class edge:

    def __init__(self):
        self.weight = None
        self.from_node = None
        self.to_node = None


class node:

    def __init__(self):
        self.value = None
        self.delta = None
        self.in_edge = []
        self.out_edge = []
        self.bias = None

    def build_in_edge(self, linked_node):
        '''builds a connection from linked_node to the node in question'''
        new_edge = edge()
        new_edge.from_node = linked_node
        new_edge.to_node = self
        self.in_edge.append(new_edge)
        linked_node.out_edge.append(new_edge)

    def build_out_edge(self, linked_node):
        new_edge = edge()
        new_edge.from_node = self
        new_edge.to_node = linked_node
        self.out_edge.append(new_edge)
        linked_node.in_edge.append(new_edge)


def activation_function(x):
    return (x)


class network:

    def __init__(self, *layers):

        self.layers = []
        self.nodes = []

        for layer_id in range(0, len(layers)):
            self.layers.append([])
            for node_id in range(0, layers[layer_id]):
                new_node = node()
                self.nodes.append(new_node)
                self.layers[layer_id].append(new_node)
                if layer_id != 0:
                    for previous_layer_node in self.layers[layer_id - 1]:
                        new_node.build_in_edge(previous_layer_node)

    def init_random_weights(self):
        for node in self.nodes:
            for edge in node.in_edge:
                if edge.weight == None:
                    edge.weight = 1
            for edge in node.out_edge:
                if edge.weight == None:
                    edge.weight = 1
            if node not in self.layers[0]:
                node.bias = 0

    def forward_propagate(self, training_data_x):
        result=[]

        for input_layer_node_id in range(0, len(self.layers[0])):
            self.layers[0][input_layer_node_id].value = training_data_x[input_layer_node_id]

        for hiddenlayer in self.layers[1:]:
            for hidden_layer_node in hiddenlayer:
                tmp_sum = 0
                for edge in hidden_layer_node.in_edge:
                    tmp_sum += edge.from_node.value * edge.weight
                tmp_sum += hidden_layer_node.bias
                hidden_layer_node.value = activation_function(tmp_sum)
        for node_id in self.layers[-1]:
            result.append(node_id.value)
        return(result)

if __name__ == '__main__':

    # trainning_data = [
    #     [0+1, 0+1],
    #     [0+1, 1+1],
    #     [1+1, 0+1],
    #     [1+1, 1+1]   ]

    n = network(3, 1)
    n.init_random_weights()
    result = n.forward_propagate([1,1,1])
    print(result)
