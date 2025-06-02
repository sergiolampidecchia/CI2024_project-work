#   *        Giovanni Squillero's GP Toolbox
#  / \       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2   +      A no-nonsense GP in pure Python
#    / \
#  10   11   Distributed under MIT License

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout

def draw(node, filename=None):
    G = nx.DiGraph()
    labels = {}
    colors = {}
    shapes = {}

    def get_display_name(name):
        replacements = {
            "add_fn": "+",
            "sub_fn": "-",
            "mul_fn": "*",
            "div_safe": "/",
            "neg_fn": "neg",
            "log_safe": "log",
            "sqrt_safe": "sqrt",
            "exp_safe": "exp",
            "abs_fn": "abs",
            "sin_fn": "sin",
            "cos_fn": "cos"
        }
        return replacements.get(name, name)

    for n1 in list(node.subtree):
        nid1 = id(n1)
        raw_name = getattr(n1, "short_name", str(n1))
        name = get_display_name(raw_name)

        if n1.is_leaf:
            if raw_name.startswith("x"):  
                labels[nid1] = name
                colors[nid1] = "lightblue"
                shapes[nid1] = "s"
            else:  
                try:
                    val = float(raw_name)
                    labels[nid1] = f"{val:.3f}"
                except:
                    labels[nid1] = name
                colors[nid1] = "lightgreen"
                shapes[nid1] = "s"
        else:  
            labels[nid1] = name
            colors[nid1] = "lightpink"
            shapes[nid1] = "o"

        for n2 in n1._successors:
            G.add_edge(nid1, id(n2))

    pos = graphviz_layout(G, prog="dot")

    fig, ax = plt.subplots(figsize=(12, 8))

    nx.draw_networkx_edges(G, pos, ax=ax)

    for shape in set(shapes.values()):
        nodes = [n for n in G.nodes if shapes[n] == shape]
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=nodes,
                               node_color=[colors[n] for n in nodes],
                               node_shape=shape, node_size=1500)

    nx.draw_networkx_labels(G, pos, ax=ax, labels=labels, font_size=10)

    ax.set_axis_off()
    plt.tight_layout()

    if filename:
        plt.savefig(filename, format="png", bbox_inches="tight")
        plt.close()
    else:
        plt.show()
