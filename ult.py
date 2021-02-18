# v -->  a -->  C
# b --> C
# b --> v -- a --> C
# v --> v ---> a -- > C
from collections import defaultdict
import random
from node import Placeholder

def toplogic(graph):
    sorted_node = []

    while len(graph) > 0:

        all_inputs = []
        all_outputs = []

        for n in graph:
            all_inputs += graph[n]
            all_outputs.append(n)

        all_inputs = set(all_inputs)
        all_outputs = set(all_outputs)

        need_remove = all_outputs - all_inputs  # which in all_inputs but not in all_outputs

        if len(need_remove) > 0:
            node = random.choice(list(need_remove))

            visited_next = [node]
            if len(graph) == 1:  visited_next += graph[node]

            graph.pop(node)
            sorted_node += visited_next

            for _, links in graph.items():
                if node in links: links.remove(node)
        else:
            break

    return sorted_node


def convert_feed_dict_to_graph(feed_dict):
    computing_graph = defaultdict(list)

    nodes = [n for n in feed_dict]

    while nodes:
        n = nodes.pop(0)
        if isinstance(n, Placeholder):
            n.value = feed_dict[n]

        if n in computing_graph:
            continue

        for m in n.outputs:
            computing_graph[n].append(m)
            nodes.append(m)

    return computing_graph


def topological_sort_feed_dict(feed_dict):
    graph = convert_feed_dict_to_graph(feed_dict)
    return toplogic(graph)


def forward(graph_order, monitor=False):
    for node in graph_order:
        if monitor: print('forward compuiting -- {}'.format(node))
        node.forward()
        # print('fv \n',node.value,node.name)


def backward(graph_order, monitor=False):
    for node in graph_order[::-1]:
        if monitor: print('backward computing -- {}'.format(node))
        node.backward()
        # print('bv \n',node.value,node.name)


def forward_and_backward(graph_order, monitor=False):
    forward(graph_order, monitor)
    backward(graph_order, monitor)
