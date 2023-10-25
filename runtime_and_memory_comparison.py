import ogb
import scipy.sparse as sp
import networkx as nx
import numpy as np
import sys
from ogb.linkproppred import LinkPropPredDataset, NodePropPredDataset
from utils import normal_equations, uniform_sampling_rows, lev_score_sampling, normalize_adj, generate_dataset, generate_dataset_syn
%load_ext memory_profiler

input = sys.argv[1]

if input == "house":
    X = np.loadtxt("house/X.csv", delimiter = ",", skiprows = 1)
    y = np.loadtxt("house/y.csv", delimiter = ",", skiprows = 1)
    A_G = nx.adjacency_matrix(nx.read_graphml("house/graph.graphml"))
    range_start = 15000
    range_end = 15200
    A_G = A_G[range_start:range_end, range_start:range_end]
    A_G = normalize_adj(A_G)
    X = X[range_start:range_end]
    X = X/np.linalg.norm(X)
    y = y[range_start:range_end]
    num_nodes = A_G.shape[0]

elif input == "ogbl-ddi":
    dataset = LinkPropPredDataset(name = "ogbl-ddi")
    graph = dataset[0] # graph: library-agnostic graph object
    num_nodes = graph["num_nodes"]
    A_G = sp.csr_array((np.ones(len(graph["edge_index"][0])), (graph["edge_index"][0], graph["edge_index"][1])), shape=(num_nodes, num_nodes))
    X, y, w = generate_dataset_syn(A_G, 1, 50, 1, 1, num_nodes, 100)

elif input == "ogbn-arxiv":
    dataset = NodePropPredDataset(name = "ogbn-arxiv" )
    graph, label = dataset[0] # graph: library-agnostic graph object
    num_nodes = graph["num_nodes"]
    A_G = sp.csr_array((np.ones(len(graph["edge_index"][0])), (graph["edge_index"][0], graph["edge_index"][1])), shape=(num_nodes, num_nodes))
    X = graph["node_feat"]
    y = label

elif input == "generated-data":
    n = [50000, 100000, 150000]
    p = 500
    num_nodes = n[0]
    %memit A_G = np.random.normal(1, 1, size=(num_nodes, num_nodes))
    A_G = sp.csr_array(A_G)
    X, y, w = generate_dataset(A_G, 1, 1, 1, 1, num_nodes, p)


sketch_size = np.ceil(np.log(num_nodes))
budget = num_nodes*sketch_size

from timeit import default_timer as timer

print("Method 1: Regression with full AX")
start1 = timer()

%memit full_comp = A_G@X
w_full = normal_equations(full_comp, y)

elapsed1 = timer()

print("Regression with full AX:", elapsed1 - start1)

print("Method 2: Regression with Uniformly Sampled (AX, y)")
start2 = timer()
    
%memit A_uni_sampled, sampled_idx = uniform_sampling_rows(A_G, budget)
%memit A_uni_sampled = A_uni_sampled @ X
w_uni = normal_equations(A_uni_sampled, y[sampled_idx])

elapsed2 = timer()

print("Regression with Uniformly Sampled (AX, y):", elapsed2 - start2)


print("Method 3: Regression via exact leverage score sampling")
start3 = timer()

eff_data_mat = A_G@X
exact_lev_score = lev_exact(eff_data_mat)

%memit sampled_rows1, exact_lev_data_mat, scaling_vec1 = lev_score_sampling(A_G, sketch_size, exact_lev_score)
%memit exact_lev_data_mat = exact_lev_data_mat @ X

scaled_labels1 = y[sampled_rows1] / scaling_vec1
w_exact_lev = normal_equations(exact_lev_data_mat, scaled_labels1)

elapsed3 = timer()

print("Regression via exact leverage score sampling:", elapsed3 - start3)


print("Method 4: Regression via approximate leverage-score sampling through uniform sampling")
start4 = timer()

A_approx, sampled_idx2 = rank_one_uniform_sampling(A_G, X, budget)
approx_lev_score = lev_approx2(A_approx, 0.5)

%memit sampled_rows2, approx_lev_data_mat, scaling_vec2 = lev_score_sampling(A_G, sketch_size, approx_lev_score)
%memit approx_lev_data_mat = approx_lev_data_mat@X

scaled_labels2 = y[sampled_rows2] / scaling_vec2
w_approx_lev = normal_equations(approx_lev_data_mat, scaled_labels2)

elapsed4 = timer()

print("Regression via approximate leverage-score sampling through uniform sampling:", elapsed4 - start4)

print("Method 5: Regression via approximate leverage-score sampling with minVar")
start5 = timer()

A_approx_minVar, sampled_idx3 = rank_one_minVar_sampling(A_G, X, budget)
approx_lev_score_minVar = lev_approx2(A_approx_minVar, 0.5)

%memit sampled_rows3, approx_lev_data_mat_minVar, scaling_vec3 = lev_score_sampling(A_G, sketch_size, approx_lev_score_minVar)
%memit approx_lev_data_mat_minVar = approx_lev_data_mat_minVar@X

scaled_labels3 = y[sampled_rows3] / scaling_vec3
w_approx_lev_minVar = normal_equations(approx_lev_data_mat_minVar, scaled_labels3)

elapsed5 = timer()

print("Regression via approximate leverage-score sampling with minVar:", elapsed5 - start5)