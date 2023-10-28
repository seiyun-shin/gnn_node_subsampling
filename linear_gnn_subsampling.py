from copy import deepcopy
from ogb.linkproppred import LinkPropPredDataset
from ogb.nodeproppred import NodePropPredDataset
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from timeit import default_timer as timer
from utils import normal_equations, normalize_adj, uniform_sampling_rows, rank_one_uniform_sampling, rank_one_minVar_sampling,\
    lev_exact, lev_approx2, lev_score_sampling, GraphSage, GraphSaint, generate_dataset, generate_dataset_syn,\
    run_GCN_linear
import ogb
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp
import sklearn
import sys
import torch
import torch_geometric.utils as pygutils
import torch.optim as optim


error_msg = "Usage: python " + __file__ + \
    "(dataset: house|ogbl-ddi|ogbn-arxiv|generated-data|facebook)"
if len(sys.argv) != 2:
    print(error_msg)
    sys.exit()
input = sys.argv[1]

if input == "house":
    X = np.loadtxt("dataset/house/X.csv", delimiter=",", skiprows=1)
    y = np.loadtxt("dataset/house/y.csv", delimiter=",", skiprows=1)
    A_G = nx.adjacency_matrix(nx.read_graphml("dataset/house/graph.graphml"))
    range_start = 15500
    range_end = 16500
    A_G = A_G[range_start:range_end, range_start:range_end]
    A_G = A_G.toarray()
    X = X[range_start:range_end]
    X = X/np.linalg.norm(X)
    y = y[range_start:range_end]
    num_nodes = A_G.shape[0]

elif input == "ogbl-ddi":
    dataset = LinkPropPredDataset(name="ogbl-ddi")
    graph = dataset[0]  # graph: library-agnostic graph object
    num_nodes = graph["num_nodes"]
    A_G = sp.csr_array((np.ones(len(graph["edge_index"][0])), (
        graph["edge_index"][0], graph["edge_index"][1])), shape=(num_nodes, num_nodes))
    A_G = A_G.toarray()
    X, y, w = generate_dataset_syn(A_G, 1, 50, 1, 1, num_nodes, 100)

elif input == "ogbn-arxiv":
    dataset = NodePropPredDataset(name="ogbn-arxiv")
    graph, label = dataset[0]  # graph: library-agnostic graph object
    num_nodes = graph["num_nodes"]
    A_G = sp.csr_array((np.ones(len(graph["edge_index"][0])), (
        graph["edge_index"][0], graph["edge_index"][1])), shape=(num_nodes, num_nodes))
    A_G = A_G.toarray()
    X = graph["node_feat"]
    y = label

elif input == "generated-data":
    num_nodes = [50000, 100000, 150000]
    p = 500
    num_nodes = num_nodes[0]
    A_G = np.random.normal(1, 1, size=(num_nodes, num_nodes))
    X, y, w = generate_dataset(A_G, 1, 1, 1, 1, num_nodes, p)

elif input == "facebook":
    B = np.zeros((4039, 4039))
    f = open("dataset/facebook/facebook_combined.txt", 'r')
    while True:
        line = f.readline()
        if not line:
            break
        edge_string = line.split()
        edge = np.array(list(map(int, edge_string)))
        B[edge[0], edge[1]] = 1
    f.close()
    A_G = sp.csr_array(np.maximum(B, B.T))
    num_nodes = A_G.shape[0]
    A_G = A_G.toarray()
    X, y, w = generate_dataset_syn(A_G, 1, 10, 1, 10, num_nodes, 100)
    X = X/np.linalg.norm(X)

else:
    print(error_msg)
    sys.exit()


# '''Type 3 plot MSE & R2 score w.r.t. budget'''
'''Hyperparameters'''
hidden_dim = 32
epochs = 300
num_trials = 10
tot_budget_vec = np.array(
    [0.003, 0.005, 0.007, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7])

'''Type 3 plot MSE & R2 score w.r.t. budget'''
data_mat = X
labels = y


sketch_size_vec = np.ceil(np.multiply(tot_budget_vec, num_nodes))
print(sketch_size_vec)
budget_vec = np.multiply(tot_budget_vec, 1)
len_budget = len(budget_vec)
budget_vec_percent = np.multiply(tot_budget_vec, 100)
budget_vec = np.ceil(np.multiply(budget_vec, (num_nodes ** 2)))
idx_vec = np.arange(0, len_budget, 1)
print(idx_vec)

tot_budget_vec = np.ceil(np.multiply(tot_budget_vec, (num_nodes ** 2)))
tot_budget_vec = tot_budget_vec.astype(int)

MSE_OLS_mat = np.zeros((num_trials, len_budget))
MSE_Uniform_mat = np.zeros((num_trials, len_budget))
MSE_ExactLev_mat = np.zeros((num_trials, len_budget))
MSE_ApproxLev_mat = np.zeros((num_trials, len_budget))
MSE_ApproxLev_minVar_mat = np.zeros((num_trials, len_budget))
MSE_GraphSage_mat = np.zeros((num_trials, len_budget))
MSE_GraphSaint_mat = np.zeros((num_trials, len_budget))


s_adj = pygutils.dense_to_sparse(torch.from_numpy(A_G))
edge_index = s_adj[0]
edge_weight = s_adj[1]
test_mask = np.full(num_nodes, True)
MSE_OLS = run_GCN_linear(data_mat, edge_index, edge_weight, labels, test_mask,
                         data_mat, edge_index, edge_weight, labels, hidden_dim, epochs)

'''compute the exact leverage score'''
exact_lev_score = lev_exact(A_G @ data_mat)

sketch_size_vec_original = np.ceil(np.multiply(tot_budget_vec, num_nodes))
budget_vec_original = np.multiply(tot_budget_vec, 1)

# Assign variables to the y-axis part of the curve
for trial in np.arange(num_trials):
    print("Trial : " + str(trial+1))

    for idx in idx_vec:
        print("Budget : " + str(idx) + " " +
              str(sketch_size_vec[idx]/num_nodes))
        print(
            "------------------------------------------------------------------------------")
        #  1. GCN with exact AX------------------------------------------------------------------------------
        MSE_OLS_mat[trial, idx] = MSE_OLS

        #  2. UNI SAMPLED GCN------------------------------------------------------------------------------
        A_uni_sampled, sampled_idx = uniform_sampling_rows(
            A_G, tot_budget_vec[idx])
        s_adj = pygutils.dense_to_sparse(torch.from_numpy(A_uni_sampled))
        uni_edge_index = s_adj[0]
        uni_edge_weight = s_adj[1]
        uni_labels = np.zeros(num_nodes)
        test_mask = np.full(num_nodes, False)
        for i in sampled_idx:
            test_mask[i] = True
            uni_labels[i] = labels[i]

        MSE_Uniform_mat[trial, idx] = run_GCN_linear(data_mat, uni_edge_index, uni_edge_weight,
                                                     uni_labels, test_mask, data_mat, edge_index, edge_weight, labels, hidden_dim, epochs)

        #  3. EXACT LEV-SCORE SAMPLED GCN------------------------------------------------------------------------------
        sampled_rows1, exact_lev_A_G, scaling_vec1 = lev_score_sampling(
            A_G, sketch_size_vec[idx], exact_lev_score)
        scaled_labels1 = np.zeros(num_nodes)
        for i in range(len(sampled_rows1)):
            scaled_labels1[sampled_rows1[i]
                           ] = labels[sampled_rows1[i]]/scaling_vec1[i]

        s_adj = pygutils.dense_to_sparse(torch.from_numpy(exact_lev_A_G))
        exact_lev_edge_index = s_adj[0]
        exact_lev_edge_weight = s_adj[1]
        test_mask = np.full(num_nodes, False)
        for i in sampled_rows1:
            test_mask[i] = True

        MSE_ExactLev_mat[trial, idx] = run_GCN_linear(data_mat, exact_lev_edge_index, exact_lev_edge_weight,
                                                      scaled_labels1, test_mask, data_mat, edge_index, edge_weight, labels, hidden_dim, epochs)

        #  4. APPROX LEV-SCORE SAMPLED GCN------------------------------------------------------------------------------
        A_approx, sampled_idx2 = rank_one_uniform_sampling(
            A_G, data_mat, budget_vec[idx])
        approx_lev_score = lev_approx2(A_approx, 0.6)
        sampled_rows2, approx_lev_A_G, scaling_vec2 = lev_score_sampling(
            A_G, sketch_size_vec[idx], approx_lev_score)

        scaled_labels2 = np.zeros(num_nodes)
        for i in range(len(sampled_rows2)):
            scaled_labels2[sampled_rows2[i]
                           ] = labels[sampled_rows2[i]]/scaling_vec2[i]

        s_adj = pygutils.dense_to_sparse(torch.from_numpy(approx_lev_A_G))
        approx_lev_edge_index = s_adj[0]
        approx_lev_edge_weight = s_adj[1]
        test_mask = np.full(num_nodes, False)
        for i in sampled_rows2:
            test_mask[i] = True

        MSE_ApproxLev_mat[trial, idx] = run_GCN_linear(data_mat, approx_lev_edge_index, approx_lev_edge_weight,
                                                       scaled_labels2, test_mask, data_mat, edge_index, edge_weight, labels, hidden_dim, epochs)

        #  5. APPROX LEV-SCORE MINVAR SAMPLED GCN------------------------------------------------------------------------------
        A_approx_minVar, sampled_idx3 = rank_one_minVar_sampling(
            A_G, data_mat, budget_vec[idx])
        approx_lev_score_minVar = lev_approx2(A_approx_minVar, 0.6)
        sampled_rows3, approx_lev_A_G_minVar, scaling_vec3 = lev_score_sampling(
            A_G, sketch_size_vec[idx], approx_lev_score_minVar)

        scaled_labels3 = np.zeros(num_nodes)
        for i in range(len(sampled_rows3)):
            scaled_labels3[sampled_rows3[i]
                           ] = labels[sampled_rows3[i]]/scaling_vec3[i]

        s_adj = pygutils.dense_to_sparse(
            torch.from_numpy(approx_lev_A_G_minVar))
        approx_lev_minvar_edge_index = s_adj[0]
        approx_lev_minvar_edge_weight = s_adj[1]
        test_mask = np.full(num_nodes, False)
        for i in sampled_rows3:
            test_mask[i] = True

        MSE_ApproxLev_minVar_mat[trial, idx] = run_GCN_linear(data_mat, approx_lev_minvar_edge_index, approx_lev_minvar_edge_weight,
                                                              scaled_labels3, test_mask, data_mat, edge_index, edge_weight, labels, hidden_dim, epochs)

        #  6. GraphSage based sampling + GCN------------------------------------------------------------------------------
        A_GraphSage, A_mask, col_idx_list = GraphSage(A_G, budget_vec[idx])
        sage_labels = np.zeros(num_nodes)
        s_adj_sg = pygutils.dense_to_sparse(torch.from_numpy(A_GraphSage))
        GraphSage_edge_index = s_adj_sg[0]
        GraphSage_edge_weight = s_adj_sg[1]
        test_mask = np.full(num_nodes, False)
        for i in col_idx_list:
            test_mask[i] = True
            sage_labels[i] = labels[i]

        MSE_GraphSage_mat[trial, idx] = run_GCN_linear(data_mat, GraphSage_edge_index, GraphSage_edge_weight,
                                                       sage_labels, test_mask, data_mat, edge_index, edge_weight, labels, hidden_dim, epochs)

        #  7. GraphSaint based sampling + GCN------------------------------------------------------------------------------
        A_GraphSaint, X_GraphSaint, sampled_idx5 = GraphSaint(
            A_G, data_mat, budget_vec[idx])
        saint_labels = np.zeros(num_nodes)
        for i in sampled_idx5:
            saint_labels[i] = labels[i]
        s_adj = pygutils.dense_to_sparse(torch.from_numpy(A_GraphSaint))
        GraphSaint_edge_index = s_adj[0]
        GraphSaint_edge_weight = s_adj[1]
        test_mask = np.full(num_nodes, True)

        MSE_GraphSaint_mat[trial, idx] = run_GCN_linear(X_GraphSaint, GraphSaint_edge_index, GraphSaint_edge_weight,
                                                        saint_labels, test_mask, data_mat, edge_index, edge_weight, labels, hidden_dim, epochs)

# Average over the number of trials
'''1. MSE via full AX'''
mean_MSE_OLS_vec = np.mean((MSE_OLS_mat), axis=0)
std_MSE_OLS_vec = 1.96*np.std((MSE_OLS_mat), axis=0)/np.sqrt(num_trials)

'''2. MSE via uniform sampling'''
mean_MSE_Uniform_vec = np.mean((MSE_Uniform_mat), axis=0)
std_MSE_Uniform_vec = 1.96 * \
    np.std((MSE_Uniform_mat), axis=0)/np.sqrt(num_trials)
# print(std_MSE_Uniform_vec)

'''3. MSE via exact leverage-score sampling'''
mean_MSE_ExactLev_vec = np.mean((MSE_ExactLev_mat), axis=0)
std_MSE_ExactLev_vec = 1.96 * \
    np.std((MSE_ExactLev_mat), axis=0)/np.sqrt(num_trials)

'''4. MSE via Algorithm 1'''
mean_MSE_ApproxLev_vec = np.mean((MSE_ApproxLev_mat), axis=0)
std_MSE_ApproxLev_vec = 1.96 * \
    np.std((MSE_ApproxLev_mat), axis=0)/np.sqrt(num_trials)

'''5. MSE via Algorithm 2'''
mean_MSE_ApproxLev_minVar_vec = np.mean((MSE_ApproxLev_minVar_mat), axis=0)
std_MSE_ApproxLev_minVar_vec = 1.96 * \
    np.std((MSE_ApproxLev_minVar_mat), axis=0)/np.sqrt(num_trials)

'''6. MSE via GraphSage based sampling'''
mean_MSE_GraphSage_vec = np.mean((MSE_GraphSage_mat), axis=0)
std_MSE_GraphSage_vec = 1.96 * \
    np.std((MSE_GraphSage_mat), axis=0)/np.sqrt(num_trials)

'''7. MSE via GraphSaint based sampling'''
mean_MSE_GraphSaint_vec = np.mean((MSE_GraphSaint_mat), axis=0)
std_MSE_GraphSaint_vec = 1.96 * \
    np.std((MSE_GraphSaint_mat), axis=0)/np.sqrt(num_trials)

# Plotting all the curves simultaneously
plt.errorbar(budget_vec_percent, mean_MSE_OLS_vec, yerr=std_MSE_OLS_vec,
             color='tab:green', label='MSE with full $AX$')
plt.errorbar(budget_vec_percent, mean_MSE_Uniform_vec, yerr=std_MSE_Uniform_vec,
             color='tab:purple', label='MSE with uniformly sampled $AX$')
plt.errorbar(budget_vec_percent, mean_MSE_ExactLev_vec, yerr=std_MSE_ExactLev_vec,
             color='tab:blue', label='MSE via exact leverage score sampling of $AX$')
plt.errorbar(budget_vec_percent, mean_MSE_ApproxLev_vec,
             yerr=std_MSE_ApproxLev_vec, color='tab:red', label='MSE via Algorithm 1')
plt.errorbar(budget_vec_percent, mean_MSE_ApproxLev_minVar_vec,
             yerr=std_MSE_ApproxLev_minVar_vec, color='tab:orange', label='MSE via Algorithm 2')
plt.errorbar(budget_vec_percent, mean_MSE_GraphSage_vec,
             yerr=std_MSE_GraphSage_vec, color='tab:cyan', label='MSE via GraphSage')
plt.errorbar(budget_vec_percent, mean_MSE_GraphSaint_vec,
             yerr=std_MSE_GraphSaint_vec, color='tab:grey', label='MSE via GraphSaint')


# Naming the x-axis, y-axis and set the title
plt.xlabel("Budget (%)")
plt.ylabel("MSE")
plt.yscale("log")
plt.legend(loc='upper right')
plt.savefig('results/linear_'+input+'.svg')
plt.show()
