# Efficient Learning of Linear Graph Neural Networks via Node Subsampling (NeurIPS 2023)

https://neurips.cc/virtual/2023/poster/70314

## Summary



## Dependency

Please install dependencies by `pip install -r requirements.txt`.

[PyTorch](https://pytorch.org/) may not be installed due to GPU issues. Then please install it from the [official website](https://pytorch.org/) (e.g., by cpuonly flag).

## Files

* `runtime_and_memory_comparison.py` should run with memory_profiler and reproduces computation time and peak memory comparison among full AX .
* `linear_gnn_subsampling.py` reproduces linear mse on datasets: house, ogbl-ddi, ogbn-arxiv, facebook, or generated datasets.
* `nonlinear_gnn_subsampling.py` reproduces nonlinear mse on datasets: house, ogbl-ddi, ogbn-arxiv, or generated datasets.
* `utils.py` defines GCN model, sampling methods,  when evaluating different datasets.

## Datasets

* house: https://github.com/nd7141/bgnn
* ogbl-ddi: https://ogb.stanford.edu/docs/linkprop/#ogbl-ddi
* ogbn-arxiv: https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv
* facebook: https://snap.stanford.edu/data/ego-Facebook.html
* generated-data: 


## Evaluation

Reproduce the results by the following commands.

```
$ python memory_profiler -m runtime_and_memory_comparison.py (dataset: house|ogbl-ddi|ogbn-arxiv|generated-data)
$ python linear_gnn_subampling.py (dataset: house|ogbl-ddi|ogbn-arxiv|generated-data)
$ python nonlinear_gnn_subsampling.py (dataset: house|ogbl-ddi|ogbn-arxiv|generated-data|facebook)
```

The results are saved in `results` directiory.

### Results



## Citation

```
@inproceedings{shin2023efflineargnn,
  title={Efficient Learning of Linear Graph Neural Networks via Node Subsampling},
  author={Shin, Seiyun and Zhao, Han and Shomorony, Ilan},
  booktitle={Proceedings of the 37th Advances in Neural Information Processing Systems (NeurIPS)},
  year={2023}
}
```
