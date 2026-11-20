import logging
import numpy as np
import tqdm

from multiprocessing import Pool, cpu_count
from grakel import GraphKernel
from omegaconf import OmegaConf
from fgutils.chem.ps import atomic_num2sym
from torch_geometric.data import Data, Batch
from sklearn.utils import Bunch
from sklearn.utils import gen_batches
from grakel.graph import Graph
from src.metric import get_metric_dict


def torch2grakel(
    data, node_attr_transform=None, edge_attr_transform=None
) -> Graph | Bunch:
    def _convert_sample(data: Data, nat_fn, eat_fn) -> Graph:
        assert data.edge_index is not None
        assert data.x is not None
        edges = [(e[0], e[1]) for e in data.edge_index.T.tolist()]
        node_attr = {i: nat_fn(e) for i, e in enumerate(data.x.tolist())}
        edge_attr = None
        if data.edge_attr is not None:
            edge_attr = {k: eat_fn(e) for k, e in zip(edges, data.edge_attr.tolist())}
        g = Graph(
            edges, node_labels=node_attr, edge_labels=edge_attr, graph_format="all"
        )
        return g

    if node_attr_transform is None:
        node_attr_transform = lambda x: x
    if edge_attr_transform is None:
        edge_attr_transform = lambda x: x

    if isinstance(data, Data):
        return _convert_sample(data, node_attr_transform, edge_attr_transform)
    elif isinstance(data, Batch):
        raise NotImplementedError("Batch is not implemented.")
    else:
        graphs = [
            _convert_sample(e, node_attr_transform, edge_attr_transform) for e in data
        ]
        targets = [float(e.y[0].item()) for e in data]
        return Bunch(data=graphs, target=targets)


def to_svm_dataset(dataset):
    def attr2label(x):
        return atomic_num2sym[int(x[0])]

    svm_ds = torch2grakel(dataset, node_attr_transform=attr2label)
    return svm_ds


def init_kernel(cfg):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    kernel_name = cfg_dict.pop("type")
    normalize = cfg_dict.pop("normalize", False)
    cfg_dict.pop("n_jobs", None)
    _params = {"name": kernel_name, **cfg_dict}
    logging.info("Init kernel '{}': {}".format(kernel_name, _params))
    kernel = GraphKernel([_params], n_jobs=None, normalize=normalize)
    return kernel


_kernel = None
_svm = None
_X_test = None


def init_worker(kernel, svm, X_test):
    global _kernel, _svm, _X_test
    _kernel = kernel
    _svm = svm
    _X_test = X_test


def predict_batch(batch_slice):
    assert _kernel is not None
    assert _X_test is not None
    assert _svm is not None
    _X = _X_test[batch_slice]
    _K = _kernel.transform(_X)
    return _svm.predict_proba(_K)


def test_model(
    kernel, svm, X_test, y_test, batch_size=10, return_prediction=False, progress=False
):
    batches = list(gen_batches(len(X_test), batch_size))

    with Pool(
        processes=cpu_count(),
        initializer=init_worker,
        initargs=(kernel, svm, X_test),
    ) as pool:
        logging.info(f"Run pool with {pool._processes} processes.")
        if progress:
            pred_chunks = [None] * len(batches)
            with tqdm.tqdm(total=len(batches), desc="Predicting", unit="batch") as pbar:
                for i, result in enumerate(pool.imap(predict_batch, batches)):
                    pred_chunks[i] = result
                    pbar.update(1)
        else:
            pred_chunks = pool.map(predict_batch, batches)

    pred_test = np.vstack(pred_chunks)
    test_result = get_metric_dict(y_test, pred_test)
    if return_prediction:
        return test_result, pred_test
    else:
        return test_result
