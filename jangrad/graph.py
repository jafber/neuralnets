from graphviz import Digraph
from value import Value


def traverse(dot: Digraph, node: Value, vis: set = None):
    parent = str(id(node))
    if vis:
        if parent in vis:
            return parent
    else:
        vis = set()
    vis.add(parent)
    dot.node(name=parent,
             label=f'{{ {node.label or id(node)} | d={node.data} | g={node.grad} }}', shape='record')
    if node.children:
        children = [traverse(dot, child, vis) for child in node.children]
        op = 'op_'+parent
        dot.node(name=op, label=node.op)
        for child in children:
            dot.edge(child, op)
        dot.edge(op, parent)
    return parent


def draw(root: Value):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
    traverse(dot, root)
    return dot
