import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import cauchy
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim):
        super().__init__()
        hidden_dim = hidden_dim
        output_dim = 1
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin1 = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lin1(x)

        return x


class GCN_linear(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim):
        super().__init__()
        hidden_dim = hidden_dim
        output_dim = 1
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin1 = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = self.conv1(x, edge_index, edge_weight)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.dropout(x, training=self.training)
        x = self.lin1(x)

        return x


# Graph-weighted Linear Regression (alternative to GCN_linear)


def normal_equations(X, y):
    n = X.shape[0]
    if np.size(X) != n:
        p = X.shape[1]
    else:
        p = 1

    if p == 1:
        w = 1/(X.T @ X) * (X.T @ y)
    else:
        w = (np.linalg.pinv(X.T @ X)) @ (X.T @ y)
    return w


# UNIFORM SAMPLING


def uniform_sampling_rows(A_mat, budget):
    '''
    Uniform Sampling of rows on the graph adjacency matrix

    Input
        A_mat: n-by-n adjacency matrix
        budget: number of total sampled entries (observation)

    Output
        A_partial: n-by-n partially observed adjacency matrix
    '''

    n = A_mat.shape[0]
    num_rows = np.floor(min(budget, n ** 2)/n)
    num_rows = num_rows.astype(int)
    A_mask = np.zeros((A_mat.shape[0], A_mat.shape[1]))
    sampled_row_list = list(np.random.choice(
        np.arange(n), size=num_rows, replace=False))

    for idx in sampled_row_list:
        A_mask[idx, :] = np.ones(n)
    A_partial = np.multiply(A_mat, A_mask)
    return A_partial, sampled_row_list


# ALGORITHM 1


def rank_one_uniform_sampling(A_mat, data_mat, budget):
    '''
    Algorithm 1:
    Uniform Sampling to sample rank-1 matrix comprising A_mat*data_mat

    Sampling index "idx" with probabilty budget/n^2 to construct A_mat[:, idx] @ data_mat[idx, :] and sum

    Input
        A_mat: n-by-n adjacency matrix
        data_mat: n-by-d data matrix
        budget: number of total sampled entries (observation)

    Output
        A_partial: n-by-n partially observed adjacency matrix
    '''

    n = A_mat.shape[0]
    p = data_mat.shape[1]
    num_cols = np.ceil(min(budget, n ** 2)/n) + 1
    num_cols = num_cols.astype(int)
    AX_partial = np.zeros((n, p))
    sampled_col_list = list(np.random.choice(
        np.arange(n), size=num_cols, replace=False))

    for idx in sampled_col_list:
        AX_partial += A_mat[:, idx].reshape((n, -1)
                                            ) @ data_mat[idx, :].reshape((-1, p))
    AX_partial = np.multiply(n/num_cols, AX_partial)
    return AX_partial, sampled_col_list


# ALGORITHM 2


def rank_one_minVar_sampling(A_mat, data_mat, budget):
    '''
    Algorithm 2:
    Data-dependent Sampling to sample rank-1 matrix comprising A_mat*data_mat

    Sampling index "idx" with probabilty with
    (budget/n)*\|A_mat[:, idx]\|*\|data_mat[idx, :]\|/(\sum_idx \|A_mat[:, idx]\|*\|data_mat[idx, :]\|)\

    to construct A_mat[:, idx] @ data_mat[idx, :]

    Input
        A_mat: n-by-n adjacency matrix
        data_mat: n-by-p data matrix
        budget: number of total sampled entries (observation)

    Output
        A_partial: n-by-n partially observed adjacency matrix
    '''

    n = A_mat.shape[0]
    p = data_mat.shape[1]
    num_cols = min(budget, n ** 2)/n + 1
    num_cols = num_cols.astype(int)
    AX_partial = np.zeros((n, p))
    sampling_prob = np.zeros(n)

    for idx in range(n):
        sampling_prob[idx] = np.linalg.norm(
            A_mat[:, idx])*np.linalg.norm(data_mat[idx, :]) + 10e-6

    sampling_prob /= np.sum(sampling_prob)
    sampled_col_list = list(np.random.choice(
        np.arange(n), size=num_cols, p=sampling_prob, replace=False))

    for idx in sampled_col_list:
        AX_partial += np.multiply(1/sampling_prob[idx], A_mat[:, idx].reshape(
            (n, -1)) @ data_mat[idx, :].reshape((-1, p)))

    return AX_partial, sampled_col_list


# LEVERAGE SCORE CALCULATION & SAMPLING


def lev_exact(A_mat):
    '''
    Compute Exact Leverage Scores

    Input
        A := A_mat @ data_mat: n-by-p dense matrix A.

    Output
        lev_vec: n-dim vector containing the exact leverage scores
    '''

    A = A_mat
    n = A.shape[0]
    if np.size(A) != n:
        p = A.shape[1]
    else:
        p = 1
    lev_vec = np.zeros(n)
    if p == 1:
        norm_const = np.sum(A ** 2)
        for idx in range(n):
            lev_vec[idx] = (A[idx] ** 2)/norm_const
    else:
        U_mat, _, _ = np.linalg.svd(A, full_matrices=False)
        lev_vec = np.sum(U_mat ** 2, axis=1)
    return lev_vec


def lev_approx(A_mat, data_mat, eps):
    '''
    Compute Approximate Leverage Scores

    Input
        A := A_mat @ data_mat: n-by-p dense matrix A.
        eps: some slack to get a high-probabiltiy theoretical guarantee

    Output
        lev_vec: n-dim vector containing the exact leverage scores
    '''

    A = A_mat @ data_mat
    n = A.shape[0]
    if np.size(A) != n:
        p = A.shape[1]
    else:
        p = 1
    lev_vec = np.zeros(n)
    if p == 1:
        for idx in range(n):
            norm_const = np.sum(A ** 2)
            lev_vec[idx] = (A[idx] ** 2)/norm_const
    else:
        U_mat, _, _ = np.linalg.svd(A, full_matrices=False)
        lev_vec = np.sum(U_mat ** 2, axis=1)
    lev_vec *= ((1+eps) ** 2)/((1-eps) ** 2)
    return lev_vec


def lev_approx2(A_mat, eps):
    '''
    Compute Approximate Leverage Scores

    Input
        A := A_mat @ data_mat: n-by-p dense matrix A.
        eps: some slack to get a high-probabiltiy theoretical guarantee

    Output
        lev_vec: n-dim vector containing the exact leverage scores
    '''

    A = A_mat
    n = A.shape[0]
    if np.size(A) != n:
        p = A.shape[1]
    else:
        p = 1
    lev_vec = np.zeros(n)
    if p == 1:
        for idx in range(n):
            norm_const = np.sum(A ** 2)
            lev_vec[idx] = (A[idx] ** 2)/norm_const
    else:
        U_mat, _, _ = np.linalg.svd(A, full_matrices=False)
        lev_vec = np.sum(U_mat ** 2, axis=1)
    lev_vec *= ((1+eps) ** 2)/((1-eps) ** 2)
    return lev_vec


def lev_score_sampling(A_mat, sketch_size, prob_vec):
    '''
    Random Sampling according to a Given Distribution

    Input
        A_mat: n-by-p dense matrix A;
        sketch_size: sketch size;
        prob_vec: n-dim vector, containing the sampling probabilities.

    Output
        idx_vec: n-dim vector containing the indices sampled from {1, 2, ..., n};
        C_mat: n-by-s sketch containing scaled columns of A.
    '''

    n = A_mat.shape[0]
    if np.size(A_mat) != n:
        p = A_mat.shape[1]
    else:
        p = 1
    prob_vec /= np.sum(prob_vec)
    sketch_size = sketch_size.astype(int)
    idx_vec = np.random.choice(n, sketch_size, replace=False, p=prob_vec)
    scaling_vec = np.sqrt(sketch_size * prob_vec[idx_vec]) + 1e-10
    if p == 1:
        C_mat = A_mat[idx_vec] / scaling_vec
    else:
        C_mat = A_mat[idx_vec, :] / scaling_vec.reshape(len(scaling_vec), 1)
    C_mat = np.asarray(C_mat)
    return idx_vec, C_mat, scaling_vec


# A SAMPLING SCHEME INSPIRED BY GRAPHSAGE


def GraphSage(A_mat, budget):
    '''
    This is a sampling scheme inspired by GraphSage algorithm that selects the neighborhood of feature aggregation for each node.

    Input
        A_mat: n-by-n adjacency matrix
        budget: number of total sampled entries (observation)

    Output
        A_partial: n-by-n partially observed adjacency matrix
    '''

    n = A_mat.shape[0]
    num_cols = np.floor(min(budget, n ** 2)/n)
    num_cols = num_cols.astype(int)
    col_list = []
    A_mask = np.zeros((A_mat.shape[0], A_mat.shape[1]))

    for idx1 in range(n):
        sampled_col_list = list(np.random.choice(
            np.arange(n), size=num_cols, replace=False))
        col_list += sampled_col_list
        for idx2 in sampled_col_list:
            A_mask[idx1, idx2] = 1

    A_partial = np.multiply(A_mat, A_mask)
    rand_list = list(np.random.choice(
        np.arange(n), size=num_cols, replace=False))
    rand_list = np.random.choice(col_list, size=num_cols, replace=False)
    return A_partial, A_mask, rand_list


#  A SAMPLING SCHEME INSPIRED BY GRAPHSAINT


def GraphSaint(A_mat, data_mat, budget):
    '''
    This is a sampling scheme inspired by GraphSaint algorithm that selects a subgraph

    Input
        A_mat: n-by-n adjacency matrix
        budget: number of total sampled entries (observation)

    Output
        A_partial: n-by-n partially observed adjacency matrix
    '''

    n = A_mat.shape[0]
    p = data_mat.shape[1]
    num_cols = np.floor(min(budget, n ** 2)/n)
    num_cols = num_cols.astype(int)
    A_mask = np.zeros((n, n))
    X_mask = np.zeros((n, p))
    sampling_prob = np.zeros(n)

    for idx in range(n):
        sampling_prob[idx] = (np.linalg.norm(A_mat[:, idx]) ** 2) + 10e-6

    sampling_prob /= np.sum(sampling_prob)
    sampled_list = list(np.random.choice(
        np.arange(n), size=num_cols, p=sampling_prob, replace=False))

    for idx1 in sampled_list:
        for idx2 in sampled_list:
            A_mask[idx1][idx2] = 1
        X_mask[idx1, :] = 1

    A_subgraph = np.multiply(A_mat, A_mask)
    X_subgraph = np.multiply(data_mat, X_mask)
    return A_subgraph, X_subgraph, sampled_list


# DATA HANDLING


def normalize_adj(A_G):
    num_nodes = A_G.shape[0]
    diag_degree = sp.lil_array((num_nodes, num_nodes))
    rows, cols = A_G.nonzero()[0], A_G.nonzero()[1]
    for row, col in zip(rows, cols):
        diag_degree[row, row] += 1
    diag_degree = np.sqrt(sp.linalg.inv(
        diag_degree.tocsc() + sp.identity(num_nodes, format='csc')))
    diag_degree = sp.csr_array(diag_degree)
    A_G = diag_degree @ (A_G + sp.identity(num_nodes)) @ diag_degree
    return A_G


def generate_dataset(A_mat, mu_X, sigma_X, mu_w, sigma_w, n, p):
    '''
    Generate a synthetic dataset (linearly-regressed label with an additive perturbation)

    Input
        A_mat: n-by-n adjacency matrix A.
        mu_X: mean of each entry in data matrix
        sigma_X: std of each entry in data matrix
        mu_w: mean of weight vector's each entry
        sigma_w: std of weight vector's each entry
        n: number of data
        p: data dimension

    Output
        X: n-by-p data matrix
        y: n-by-1 label vector
        w: p-by-1 weight vector
    '''

    # Generate X as an array of `n` samples having `p` dimension according to Gaussian distribution
    X = np.random.normal(mu_X, sigma_X, size=(n, p))
    if p == 1:
        X = np.ravel(X)
    X /= np.linalg.norm(X)

    # Generate w as an array of `p` dimension according to Gaussian distribution
    w = np.random.normal(mu_w, sigma_w, p)  # X.shape[1] in general
    w /= np.linalg.norm(w)
    
    # Generate the random error of n samples, with a random value from a normal distribution, with a standard deviation provided in the function argument
    e = np.random.randn(n) * 1000
    # Calculate `y` according to the equation discussed
    if p == 1:
        y = w * (X.T @ A_mat) + e
    else:
        y = w.T @ (X.T @ A_mat) + e
    return X, y, w


def generate_dataset_syn(A_mat, x0_X, gamma_X, x0_w, gamma_w, n, p):
    '''
    Generate a synthetic dataset (linearly-regressed label with an additive perturbation)

    Input
        A_mat: n-by-n adjacency matrix A.
        x0_X: the location parameter (indicating the location of the peak of the distribution) of each row vector in data matrix
        gamma_X: the scale parameter (specifying the half-width at half-maximum (HWHM)) of each row vector in data matrix
        x0_w: the location parameter (indicating the location of the peak of the distribution) of weight vector
        gamma_w: the scale parameter (specifying the half-width at half-maximum (HWHM)) of weight vector
        n: number of data
        p: data dimension

    Output
        X: n-by-p data matrix
        y: n-by-1 label vector
        w: p-by-1 weight vector
    '''

    # Generate X as an array of `n` samples having `p` dimension according to Cauchy distribution
    X = cauchy.rvs(loc=10*x0_X, scale=10*gamma_X, size=(n, p))
    if p == 1:
        X = np.ravel(X)

    # Generate w as an array of `p` dimension according to Cauchy distribution
    w = cauchy.rvs(loc=10*x0_w, scale=10*gamma_w, size=p)
    
    # Generate the random error of n samples, with a random value from a normal distribution, with a standard deviation provided in the function argument
    e = np.random.normal(1, 10, n)
    # Calculate `y` according to the equation discussed
    if p == 1:
        y = w * (X.T @ A_mat) + e
    else:
        y = w.T @ (X.T @ A_mat) + e
    return X, y, w


# METRICS

def MSE(A_mat, data_mat, labels, w_hat):
    '''Returns calculated value of mse'''

    p = np.size(w_hat)
    if p == 1:
        err = np.mean((labels - w_hat * (data_mat.T @ A_mat)) ** 2)
    else:
        err = np.mean((labels - w_hat.T @ (data_mat.T @ A_mat)) ** 2)
    return err


def RSE(A_mat, data_mat, labels, w_hat):
    '''Relative Squared Error'''

    p = np.size(w_hat)
    if p == 1:
        numerator = np.sum((labels - w_hat * (data_mat.T @ A_mat)) ** 2)
        denominator = np.sum(np.subtract(labels, np.mean(labels)) ** 2)
    else:
        numerator = np.sum((labels - w_hat.T @ (data_mat.T @ A_mat)) ** 2)
        denominator = np.sum(np.subtract(labels, np.mean(labels)) ** 2)

    err = numerator / denominator
    return err


# RUNNING GCN


def run_GCN(X, edge_index, edge_weight, Y, test_mask, base_X, base_edge_index, base_edge_weight, base_Y, hidden_dim, epochs):
    num_node_features = base_X.shape[1]
    train_mask = np.ones(base_X.shape[0], dtype=bool)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(num_node_features, hidden_dim).to(device)
    data = Data(x=torch.from_numpy(X.astype(np.float32)), edge_index=edge_index, edge_weight=edge_weight.float(),
                y=torch.from_numpy(Y), train_mask=train_mask).to(device)
    base_data = Data(x=torch.from_numpy(base_X.astype(np.float32)), edge_index=base_edge_index, edge_weight=base_edge_weight.float(),
                     y=torch.from_numpy(base_Y), test_mask=test_mask).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)
    loss_fn = nn.MSELoss()
    min_mse = float('inf')
    for epoch in range(epochs):
        # Train
        model.train()
        optimizer.zero_grad()
        out = model(data)
        # loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        # y_flattened = torch.unsqueeze(data.y, dim=-1)
        outs = torch.squeeze(out)
        loss = loss_fn(outs[data.train_mask], data.y[data.train_mask].float())
        loss.backward()
        optimizer.step()

        # Evaluating on Test
        test_out = model(base_data)
        test_outs = torch.squeeze(test_out)
        model.eval()
        test_mse = loss_fn(
            test_outs[base_data.test_mask], base_data.y[base_data.test_mask].float())

        mse_val = float(test_mse)
        min_mse = min(min_mse, mse_val)
        # print("Test MSE at this epoch:", mse_val)
    return min_mse


def run_GCN_linear(X, edge_index, edge_weight, Y, test_mask, base_X, base_edge_index, base_edge_weight, base_Y, hidden_dim, epochs):
    num_node_features = base_X.shape[1]
    train_mask = np.ones(base_X.shape[0], dtype=bool)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN_linear(num_node_features, hidden_dim).to(device)
    data = Data(x=torch.from_numpy(X.astype(np.float32)), edge_index=edge_index, edge_weight=edge_weight.float(),
                y=torch.from_numpy(Y), train_mask=train_mask).to(device)
    base_data = Data(x=torch.from_numpy(base_X.astype(np.float32)), edge_index=base_edge_index, edge_weight=base_edge_weight.float(),
                     y=torch.from_numpy(base_Y), test_mask=test_mask).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)
    loss_fn = nn.MSELoss()
    min_mse = float('inf')
    for epoch in range(epochs):
        # Train
        model.train()
        optimizer.zero_grad()
        out = model(data)
        # loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        # y_flattened = torch.unsqueeze(data.y, dim=-1)
        outs = torch.squeeze(out)
        loss = loss_fn(outs[data.train_mask], data.y[data.train_mask].float())
        loss.backward()
        optimizer.step()

        # Evaluating on Test
        test_out = model(base_data)
        test_outs = torch.squeeze(test_out)
        model.eval()
        test_mse = loss_fn(
            test_outs[base_data.test_mask], base_data.y[base_data.test_mask].float())
        mse_val = float(test_mse)
        min_mse = min(min_mse, mse_val)
        # print("Test MSE at this epoch:", mse_val)
    return min_mse
