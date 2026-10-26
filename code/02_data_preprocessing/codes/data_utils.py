import os
import networkx as nx


def relabel_graph(g, offset=0):
    mapping = {}
    for i, u in enumerate(sorted(g.nodes)):
        mapping[u] = i + offset
    return nx.relabel_nodes(g, mapping)


def write_to_file(data, file_type, dataset_name, dataset_dir):
    ds_root = os.path.join(dataset_dir, dataset_name)
    if not os.path.exists(ds_root):
        os.makedirs(ds_root)
    path = os.path.join(ds_root, "{}_{}.txt".format(dataset_name, file_type))
    with open(path, "w") as f:
        for entry in data:
            entry_str = str(entry).replace(" ", "")
            if isinstance(entry, tuple):
                entry_str = entry_str.strip("(").strip(")")
            elif isinstance(entry, list):
                entry_str = entry_str.strip("[").strip("]")
            f.write("{}\n".format(entry_str))


def its_to_TUData(
    its_data: list[nx.Graph],
    label_data: list[int],
    dataset_name: str,
    tu_data_dir: str,
    verbose=True,
):
    """Convert a list of ITS graphs to the TUDataset file format. The ITS
    graphs are expected to be undirected simple vertex and edge-labeled
    networkx graphs. The node and edge features should be a list or tuple of
    numerical values. The keys are 'node_features' and 'edge_features'
    respectively.

    :param its_data: List of ITS graphs.
    :param label_data: List of ITS graph labels. The target label for each ITS
        graph. This should be of same length as its_data.
    :param tu_data_dir: The directory where the files for the TUDataset are
        stored.
    :param dataset_name: The name of the new TUDataset.
    :param verbose: (optional) Flag to control print output. (Default: True)
    """
    if len(its_data) != len(label_data):
        raise ValueError(
            (
                "Argument its_data and label_data should be of same length. "
                + "Each ITS needs a label. ({} != {})"
            ).format(len(its_data), len(label_data))
        )

    def _check_attr(expected_cnt, attr, atype):
        if expected_cnt is None:
            expected_cnt = len(attr)
        elif expected_cnt != len(attr):
            raise ValueError(
                (
                    "Inconsistent number of {} features. "
                    + "Expected {} but found {}.".format()
                ).format(atype, expected_cnt, len(attr))
            )
        return expected_cnt, attr

    DS_A = []
    DS_graph_indicator = []
    DS_graph_labels = []
    DS_node_attributes = []
    DS_edge_attributes = []

    node_idx_offset = 1

    node_cnt = 0
    edge_cnt = 0
    nattr_cnt = None
    eattr_cnt = None
    graph_cnt = len(its_data)

    for graph_id, (its, label) in enumerate(zip(its_data, label_data)):
        if its.number_of_nodes() == 0:
            raise ValueError(f"Graph at index {graph_id} has no nodes.")
        if its.number_of_edges() == 0:
            raise ValueError(f"Graph at index {graph_id} has no edges.")
        its = relabel_graph(its, offset=node_idx_offset)
        node_cnt += len(its.nodes)
        edge_cnt += 2 * len(its.edges)
        node_idx_offset += len(its.nodes)
        DS_graph_labels.append(label)
        for u, d in its.nodes(data=True):
            DS_graph_indicator.append(graph_id + 1)
            nattr_cnt, nattr = _check_attr(nattr_cnt, d["node_features"], "node")
            DS_node_attributes.append(nattr)
        for u, v, d in its.edges(data=True):
            DS_A.extend([(u, v), (v, u)])
            eattr_cnt, eattr = _check_attr(eattr_cnt, d["edge_features"], "edge")
            DS_edge_attributes.extend([eattr, eattr])
    assert edge_cnt == len(DS_A)
    assert node_cnt == len(DS_graph_indicator)
    assert graph_cnt == len(DS_graph_labels)
    assert node_cnt == len(DS_node_attributes)
    assert edge_cnt == len(DS_edge_attributes)

    if verbose:
        print("Dataset stats:")
        print("  {:<15}: {}".format("Graphs", graph_cnt))
        print("  {:<15}: {}".format("Nodes", node_cnt))
        print(
            "  {:<15}: {} (undirected: {})".format("Edges", edge_cnt, int(edge_cnt / 2))
        )
        print("  {:<15}: {:.2f}".format("Avg. Nodes", node_cnt / graph_cnt))
        print("  {:<15}: {:.2f}".format("Avg. Edges", edge_cnt / graph_cnt))
        print("  {:<15}: {:d}".format("Node Features", nattr_cnt))
        print("  {:<15}: {:d}".format("Edge Features", eattr_cnt))

    write_to_file(DS_A, "A", dataset_name, tu_data_dir)
    write_to_file(DS_graph_indicator, "graph_indicator", dataset_name, tu_data_dir)
    write_to_file(DS_graph_labels, "graph_labels", dataset_name, tu_data_dir)
    write_to_file(DS_node_attributes, "node_attributes", dataset_name, tu_data_dir)
    write_to_file(DS_edge_attributes, "edge_attributes", dataset_name, tu_data_dir)

    if verbose:
        print("DONE. Converted {} ITS graphs to TUDataset.".format(graph_cnt))
