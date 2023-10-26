from copy import deepcopy
from ogb.linkproppred import LinkPropPredDataset
from ogb.nodeproppred import NodePropPredDataset
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from timeit import default_timer as timer
from utils import normal_equations, normalize_adj, uniform_sampling_rows, rank_one_uniform_sampling, rank_one_minVar_sampling,\
    lev_exact, lev_approx2, lev_score_sampling, GraphSage, GraphSaint, generate_dataset, generate_dataset_syn,\
    run_GCN
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
    "(dataset: house|ogbl-ddi|ogbn-arxiv|generated-data)"
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
    A_G = normalize_adj(A_G)
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

else:
    print(error_msg)
    sys.exit()

'''Hyperparameters'''
hidden_dim = 32
epochs = 300
num_trials = 10
tot_budget_vec = np.array(
    [0.003, 0.005, 0.007, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7])

'''Type 3 plot MSE & R2 score w.r.t. budget'''
data_mat = X
labels = y


# fix the budget for the leverage-score sampling
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
# test_mask = np.ones(len(data_mat), dtype = bool)
test_mask = np.full(num_nodes, True)
MSE_OLS = run_GCN(data_mat, edge_index, edge_weight, labels, test_mask,
                  data_mat, edge_index, edge_weight, labels, hidden_dim, epochs)

'''compute the exact leverage score'''
exact_lev_score = lev_exact(A_G @ data_mat)
# sketch_size = math.ceil(np.log(num_nodes)/(0.1 ** 2)) #fix the budget for the leverage-score sampling

# fix the budget for the leverage-score sampling
sketch_size_vec_original = np.ceil(np.multiply(tot_budget_vec, num_nodes))
budget_vec_original = np.multiply(tot_budget_vec, 1)

# Assign variables to the y-axis part of the curve
for trial in np.arange(num_trials):
    print("Trial : " + str(trial))
    #     sketch_size_vec = deepcopy(sketch_size_vec_original)
    #     budget_vec = deepcopy(budget_vec_original)
    #     print(budget_vec)

    for idx in idx_vec:
        #         sketch_size_vec[idx] += np.ceil(0.3*tot_budget_vec[idx]/num_nodes) #fix the budget for the leverage-score sampling
        #         budget_vec[idx] -= np.ceil(0.3*tot_budget_vec[idx])
        #         print("Idx=", idx)
        #         print("Sketch size vec=", sketch_size_vec)
        #         print("Budget vec", budget_vec)
        print("Budget : " + str(idx) + " " +
              str(sketch_size_vec[idx]/num_nodes))
        print(
            "------------------------------------------------------------------------------")
        '''MSE via full AX'''
        #  1. GCN with exact AX------------------------------------------------------------------------------
        MSE_OLS_mat[trial, idx] = MSE_OLS
        # R2_OLS_mat[trial, idx] = R2_OLS
        # RSE_OLS_mat[trial, idx] = RSE_OLS

        '''MSE via Uniformly sampled A + X'''
        A_uni_sampled, sampled_idx = uniform_sampling_rows(
            A_G, tot_budget_vec[idx])
        # w_uni = normal_equations(A_uni_sampled[sampled_idx, :] @ data_mat, labels[sampled_idx])
        # MSE_Uniform = mean_squared_error(labels, w_uni.T @ (data_mat.T @ A_G))

        #  2. UNI SAMPLED GCN------------------------------------------------------------------------------
        s_adj = pygutils.dense_to_sparse(torch.from_numpy(A_uni_sampled))
        uni_edge_index = s_adj[0]
        uni_edge_weight = s_adj[1]
        uni_labels = np.zeros(num_nodes)
        test_mask = np.full(num_nodes, False)
        for i in sampled_idx:
            test_mask[i] = True
            uni_labels[i] = labels[i]

        MSE_Uniform_mat[trial, idx] = run_GCN(data_mat, uni_edge_index, uni_edge_weight,
                                              uni_labels, test_mask, data_mat, edge_index, edge_weight, labels, hidden_dim, epochs)

        # form of edge index (500, 4267), (1, 300), ...  -> adjacency matrix dimension: 4267x4267
        # label -> sampling 안된 idx에 해당하는 entryvalue를 0으로 세팅 A[sample 안된 row, :]@X = 0 W = 0

        # R2_Uniform_mat[trial, idx] = r2_score(labels, w_uni.T @ (data_mat.T @ A_G))
        # RSE_Uniform = RSE(A_G, data_mat, labels, w_uni)
        # RSE_Uniform_mat[trial, idx] = RSE_Uniform

#         print("Baseline budget exploitation:", num_samples_base)

        '''MSE via exact leverage-score sampling'''
        sampled_rows1, exact_lev_A_G, scaling_vec1 = lev_score_sampling(
            A_G, sketch_size_vec[idx], exact_lev_score)
        scaled_labels1 = np.zeros(num_nodes)
        for i in range(len(sampled_rows1)):
            scaled_labels1[sampled_rows1[i]
                           ] = labels[sampled_rows1[i]]/scaling_vec1[i]

        # w_exact_lev = normal_equations(exact_lev_A_G, scaled_labels1)

#         print("Exact lev-score-sample weight vector:",w_exact_lev)
        # MSE_exact_lev = mean_squared_error(labels, w_exact_lev.T @ (data_mat.T @ A_G))

        #  3. EXACT LEV-SCORE SAMPLED GCN------------------------------------------------------------------------------
        s_adj = pygutils.dense_to_sparse(torch.from_numpy(exact_lev_A_G))
        exact_lev_edge_index = s_adj[0]
        exact_lev_edge_weight = s_adj[1]
        test_mask = np.full(num_nodes, False)
        for i in sampled_rows1:
            test_mask[i] = True
        MSE_ExactLev_mat[trial, idx] = run_GCN(data_mat, exact_lev_edge_index, exact_lev_edge_weight,
                                               scaled_labels1, test_mask, data_mat, edge_index, edge_weight, labels, hidden_dim, epochs)

        '''MSE via approximate leverage-score sampling via uniform sampling'''
#         print("Total number of observations of A_G:", budget_vec[idx])

        A_approx, sampled_idx2 = rank_one_uniform_sampling(
            A_G, data_mat, budget_vec[idx])
        approx_lev_score = lev_approx2(A_approx, 0.6)
        sampled_rows2, approx_lev_A_G, scaling_vec2 = lev_score_sampling(
            A_G, sketch_size_vec[idx], approx_lev_score)

        scaled_labels2 = np.zeros(num_nodes)
        for i in range(len(sampled_rows2)):
            scaled_labels2[sampled_rows2[i]
                           ] = labels[sampled_rows2[i]]/scaling_vec2[i]
        # w_approx_lev = normal_equations(approx_lev_data_mat, scaled_labels2)

#         print("Algorithm 2 budget exploitation:", num_samples_uni)

        # MSE_approx_lev = mean_squared_error(labels, w_approx_lev.T @ (data_mat.T @ A_G))

        #  4. APPROX LEV-SCORE SAMPLED GCN------------------------------------------------------------------------------
        s_adj = pygutils.dense_to_sparse(torch.from_numpy(approx_lev_A_G))
        approx_lev_edge_index = s_adj[0]
        approx_lev_edge_weight = s_adj[1]
        test_mask = np.full(num_nodes, False)
        for i in sampled_rows2:
            test_mask[i] = True
        MSE_ApproxLev_mat[trial, idx] = run_GCN(data_mat, approx_lev_edge_index, approx_lev_edge_weight,
                                                scaled_labels2, test_mask, data_mat, edge_index, edge_weight, labels, hidden_dim, epochs)

        '''MSE via approximate leverage-score sampling with minVar'''
        A_approx_minVar, sampled_idx3 = rank_one_minVar_sampling(
            A_G, data_mat, budget_vec[idx])
        approx_lev_score_minVar = lev_approx2(A_approx_minVar, 0.6)
        sampled_rows3, approx_lev_A_G_minVar, scaling_vec3 = lev_score_sampling(
            A_G, sketch_size_vec[idx], approx_lev_score_minVar)

        scaled_labels3 = np.zeros(num_nodes)
        for i in range(len(sampled_rows3)):
            scaled_labels3[sampled_rows3[i]
                           ] = labels[sampled_rows3[i]]/scaling_vec3[i]
        # w_approx_lev_minVar = normal_equations(approx_lev_data_mat_minVar, scaled_labels3)

#         print("Algorithm 3 budget exploitation:", num_samples_minVar)
#         print("Approximate norm-based-sample weight vector:", w_approx_norm)

        # MSE_approx_minVar = mean_squared_error(labels, w_approx_lev_minVar.T @ (data_mat.T @ A_G))

        #  5. APPROX LEV-SCORE MINVAR SAMPLED GCN------------------------------------------------------------------------------
        s_adj = pygutils.dense_to_sparse(
            torch.from_numpy(approx_lev_A_G_minVar))
        approx_lev_minvar_edge_index = s_adj[0]
        approx_lev_minvar_edge_weight = s_adj[1]
        test_mask = np.full(num_nodes, False)
        for i in sampled_rows3:
            test_mask[i] = True
        MSE_ApproxLev_minVar_mat[trial, idx] = run_GCN(data_mat, approx_lev_minvar_edge_index, approx_lev_minvar_edge_weight,
                                                       scaled_labels3, test_mask, data_mat, edge_index, edge_weight, labels, hidden_dim, epochs)

        '''MSE via GraphSage'''
        A_GraphSage, A_mask, col_idx_list = GraphSage(A_G, budget_vec[idx])
        sage_labels = np.zeros(num_nodes)

        #  6. GraphSage + GCN------------------------------------------------------------------------------
        s_adj_sg = pygutils.dense_to_sparse(torch.from_numpy(A_GraphSage))

        GraphSage_edge_index = s_adj_sg[0]
        GraphSage_edge_weight = s_adj_sg[1]
        # for i in sampled_col_idx:
        #   test_mask[i] = True
        test_mask = np.full(num_nodes, False)
        for i in col_idx_list:
            test_mask[i] = True
            sage_labels[i] = labels[i]
        # sage_mat = A_mask @ data_mat

        MSE_GraphSage_mat[trial, idx] = run_GCN(data_mat, GraphSage_edge_index, GraphSage_edge_weight,
                                                sage_labels, test_mask, data_mat, edge_index, edge_weight, labels, hidden_dim, epochs)

        '''MSE via GraphSaint'''
        A_GraphSaint, X_GraphSaint, sampled_idx5 = GraphSaint(
            A_G, data_mat, budget_vec[idx])
        saint_labels = np.zeros(num_nodes)
        for i in sampled_idx5:
            saint_labels[i] = labels[i]

        # print(np.shape(A_GraphSaint))
        # print(np.shape(X_GraphSaint))
        # print(len(sampled_idx))

        #  7. GraphSaint + GCN------------------------------------------------------------------------------
        s_adj = pygutils.dense_to_sparse(torch.from_numpy(A_GraphSaint))
        GraphSaint_edge_index = s_adj[0]
        GraphSaint_edge_weight = s_adj[1]
        # test_mask = np.full(num_nodes, False)
        # for i in sampled_idx5:
        #   test_mask[i] = True
        test_mask = np.full(num_nodes, True)
        MSE_GraphSaint_mat[trial, idx] = run_GCN(X_GraphSaint, GraphSaint_edge_index, GraphSaint_edge_weight,
                                                 saint_labels, test_mask, data_mat, edge_index, edge_weight, labels, hidden_dim, epochs)

# averaging over the number of trials
'''MSE via full AX'''
mean_MSE_OLS_vec = np.mean((MSE_OLS_mat), axis=0)
std_MSE_OLS_vec = 1.96*np.std((MSE_OLS_mat), axis=0)/np.sqrt(num_trials)

'''MSE via uniform sampling'''
mean_MSE_Uniform_vec = np.mean((MSE_Uniform_mat), axis=0)
std_MSE_Uniform_vec = 1.96 * \
    np.std((MSE_Uniform_mat), axis=0)/np.sqrt(num_trials)
# print(std_MSE_Uniform_vec)

'''MSE via exact leverage-score sampling'''
mean_MSE_ExactLev_vec = np.mean((MSE_ExactLev_mat), axis=0)
std_MSE_ExactLev_vec = 1.96 * \
    np.std((MSE_ExactLev_mat), axis=0)/np.sqrt(num_trials)

'''MSE via Algorithm 1'''
mean_MSE_ApproxLev_vec = np.mean((MSE_ApproxLev_mat), axis=0)
std_MSE_ApproxLev_vec = 1.96 * \
    np.std((MSE_ApproxLev_mat), axis=0)/np.sqrt(num_trials)

'''MSE via Algorithm 2'''
mean_MSE_ApproxLev_minVar_vec = np.mean((MSE_ApproxLev_minVar_mat), axis=0)
std_MSE_ApproxLev_minVar_vec = 1.96 * \
    np.std((MSE_ApproxLev_minVar_mat), axis=0)/np.sqrt(num_trials)

'''MSE via GraphSage'''
mean_MSE_GraphSage_vec = np.mean((MSE_GraphSage_mat), axis=0)
std_MSE_GraphSage_vec = 1.96 * \
    np.std((MSE_GraphSage_mat), axis=0)/np.sqrt(num_trials)

'''MSE via GraphSaint'''
mean_MSE_GraphSaint_vec = np.mean((MSE_GraphSaint_mat), axis=0)
std_MSE_GraphSaint_vec = 1.96 * \
    np.std((MSE_GraphSaint_mat), axis=0)/np.sqrt(num_trials)


# Plotting both the curves simultaneously
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
# plt.title("Top-{} Error plot with {}% budget".format(k, frac*100))
# plt.title("Error plot with {}% budget".format(frac*100))
plt.yscale("log")

# plt.ylim([0, 0.15])

# Adding legend, which helps us recognize the curve according to it's color
plt.legend(loc='upper right')

# display the plot
plt.savefig('results/'+'nonlinear_'+input+'.svg')
plt.show()
