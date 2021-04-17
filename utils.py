import torch
import random
import time
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import os


def dense_tensor_to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def get_edge_inf_batch(eadj, Tmatrix, edge_feature, sampled_node):
    Tmatrix = Tmatrix
    Tmatrix = Tmatrix[sampled_node, :]
    Tmatrix_col_sum = Tmatrix.sum(0)
    edge_index = (Tmatrix_col_sum == 2).nonzero().view(-1)
    return dense_tensor_to_sparse(eadj[edge_index, :][:, edge_index]), dense_tensor_to_sparse(Tmatrix[:, edge_index]), edge_feature[edge_index, :]


def minibatch_generator(inputs, batchsize):
    assert inputs is not None
    numSamples = inputs[0].shape[0]
    num_train = len(inputs[6])
    n_val_test = len(inputs[7]) + len(inputs[8])
    n_batch = n_val_test // (batchsize - num_train)
    
    n_val_batch = len(inputs[7]) // n_batch
    n_test_batch = len(inputs[8]) // n_batch

    data_list = []

    tmp_dense_adj = inputs[0].to_dense()
    tmp_dense_eadj = inputs[3].to_dense()
    tmp_dense_tmatrix = inputs[4].to_dense()
    idx_val_shuffle = inputs[7][torch.randperm(len(inputs[7]))]
    idx_test_shuffle = inputs[8][torch.randperm(len(inputs[8]))]

    for i in range(n_batch):
        idx_val_batch = idx_val_shuffle[range(i * n_val_batch, (i+1) * n_val_batch)]
        idx_test_batch = idx_test_shuffle[range(i * n_test_batch, (i+1) * n_test_batch)]
        idx_batch = torch.cat([inputs[6], idx_val_batch, idx_test_batch])
        eadj_batch, T_batch, efeature_batch = get_edge_inf_batch(tmp_dense_eadj, tmp_dense_tmatrix, inputs[5], idx_batch)
        data_list.append([dense_tensor_to_sparse(tmp_dense_adj[idx_batch][:, idx_batch]), 
                         inputs[1][idx_batch],
                         inputs[2][idx_batch],
                         eadj_batch,
                         T_batch,
                         efeature_batch,
                         torch.tensor(range(num_train)),
                         torch.tensor(range(num_train, num_train + n_val_batch))])
    return data_list


def minibatch_generator_test(inputs, batchsize):
    assert inputs is not None
    numSamples = inputs[0].shape[0]
    num_test = len(inputs[6])
    n_batch = num_test // batchsize

    data_list = []
    # pre-convert the sparse 
    tmp_dense_adj = inputs[0].to_dense()
    tmp_dense_eadj = inputs[3].to_dense()
    tmp_dense_tmatrix = inputs[4].to_dense() 

    for i in range(n_batch - 1):
        idx_batch = inputs[6][range(i*batchsize, (i+1)*batchsize)]
        eadj_batch, T_batch, efeature_batch = get_edge_inf_batch(tmp_dense_eadj, tmp_dense_tmatrix, inputs[5], idx_batch)
        data_list.append([dense_tensor_to_sparse(tmp_dense_adj[idx_batch][:, idx_batch]),
                         inputs[1][idx_batch],
                         inputs[2][idx_batch],
                         eadj_batch,
                         T_batch,
                         efeature_batch])
    # we should test every single node
    reminder_n = num_test - (n_batch - 1) * batchsize
    reminder_idx = inputs[6][range((n_batch-1)*batchsize, num_test)]
    eadj_reminder, T_reminder, efeature_reminder = get_edge_inf_batch(tmp_dense_eadj, tmp_dense_tmatrix, inputs[5], reminder_idx)
    data_list.append([dense_tensor_to_sparse(tmp_dense_adj[reminder_idx][:, reminder_idx]),
                         inputs[1][reminder_idx],
                         inputs[2][reminder_idx],
                         eadj_reminder,
                         T_reminder,
                         efeature_reminder])
    return data_list


def create_direction_feature(directed_adj, edge_names):
    '''return a numpy array, each row represents an edge, then with two dimensions to represent the direction'''
    direction_feat = np.ndarray(shape=(len(edge_names),2), dtype=int)
    for i, x in enumerate(edge_names):
        if directed_adj[x[0], x[1]] == 1:
            direction_feat[i,:]= [1,0]
        else:
            direction_feat[i,:] = [0,1]
    return direction_feat


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def accuracy_mse(output, labels):
    diff = output - labels
    return torch.sum(diff * diff) / diff.numel()


def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def encode_onehot(labels):
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def create_edge_adj_new(vertex_adj):
    '''
    create an edge adjacency matrix from vertex adjacency matrix
    '''
    np.fill_diagonal(vertex_adj, 0)
    edge_index = np.nonzero(np.triu(vertex_adj))
    num_edge = int(len(edge_index[0]))
    edge_name = [x for x in zip(edge_index[0], edge_index[1])]

    edge_adj = np.zeros((num_edge, num_edge))
    for i in range(num_edge):
        for j in range(i, num_edge):
            if len(set(edge_name[i]) & set(edge_name[j])) == 0:
                edge_adj[i, j] = 0
            else:
                edge_adj[i, j] = 1
    adj = edge_adj + edge_adj.T
    np.fill_diagonal(adj, 1)
    return sp.csr_matrix(adj), edge_name


def node_corr(adj, feat):
    """calculate edge correlation strength"""
    prod = np.dot(feat, feat.T)
    edge_feat = prod[np.nonzero(sp.triu(adj, k=1))].T
    return sp.csr_matrix(edge_feat)


def node_corr_cosine(adj, feat):
    """calculate edge cosine distance"""
    # prod = np.dot(feat, feat.T)
    distance = squareform(pdist(feat, 'cosine'))
    edge_feat = distance[np.nonzero(sp.triu(adj, k=1))]
    ret = edge_feat.reshape((len(edge_feat), 1))
    return ret


def create_transition_matrix_new(vertex_adj):
    '''create N_v * N_e transition matrix'''
    np.fill_diagonal(vertex_adj, 0)
    edge_index = np.nonzero(np.triu(vertex_adj))
    num_edge = int(len(edge_index[0]))
    edge_name = [x for x in zip(edge_index[0], edge_index[1])]

    row_index = [i for sub in edge_name for i in sub]
    col_index = np.repeat([i for i in range(num_edge)], 2)

    data = np.ones(num_edge * 2)
    T = sp.csr_matrix((data, (row_index, col_index)),
               shape=(vertex_adj.shape[0], num_edge))

    return T


def load_data(data_name='cora', train_ratio=0.03, public_splitting=False):
    # most algorithms are overfitting for the public splitting, and we recomend to use random splitting
    if public_splitting:
        print("using util function here")
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []

        for i in range(len(names)):
            with open("./data/cite_Yang/ind.{}.{}".format(data_name, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file("../data/cite_Yang/ind.{}.test.index".format(data_name))
        test_idx_range = np.sort(test_idx_reorder)

        if data_name == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        # if citeseer data we should correct the non-label nodes
        if data_name == 'citeseer':
            zero_index = np.where(np.sum(labels, axis=1) == 0)[0]
            for i in zero_index:
                random_index = np.random.randint(labels.shape[1], size=1)[0]
                # print(random_index)
                labels[i, random_index] = 1
        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y)+500)
        labels = np.array([np.where(r==1)[0][0] for r in labels])
        # create transform matrix T, dimension is num_node * num_edge in the graph
        T = create_transition_matrix(adj)
        T = sparse_mx_to_torch_sparse_tensor(T)

        # create edge adjacent matrix from node/vertex adjacent matrix
        eadj, edge_name = create_edge_adj(adj)
        eadj= sparse_mx_to_torch_sparse_tensor(normalize(eadj))

        # create edge feature, which is the vector prod of two node features
        efeatures = create_edge_random_feature(eadj.shape[0])
        efeatures =  torch.FloatTensor(efeatures.toarray())

        adj = sparse_mx_to_torch_sparse_tensor(normalize(adj + sp.eye(adj.shape[0])))
        features = torch.FloatTensor(np.array(normalize(features).todense()))

        # create edge feature dictionary
        edge_feature_dict = {}
        for i in range(len(edge_name)):
            edge_feature_dict[edge_name[i]] = efeatures[i, :]

        labels = torch.LongTensor(labels)
        idx_train=torch.LongTensor(idx_train)
        idx_val=torch.LongTensor(idx_val)
        idx_test=torch.LongTensor(idx_test)

        return T, eadj, edge_name, edge_feature_dict, adj, features, efeatures, labels, idx_train, idx_val, idx_test
     
    else:
        # """Load citation network dataset (cora only for now)"""

        print("Not using public splitting!")
        print('Loading {} dataset...'.format(data_name))
        path = "./data/cite/"
        # check if the data is already prepared in the previous iteration
        precalculated_data = path + data_name + ".pt"
        print(precalculated_data)
        exist = os.path.isfile(precalculated_data)
        if exist:
            print("data already prepared, directly loaded! ")
            Tmat, edge_adj, edge_name, edge_feature_dict, adj, features, edge_features, labels = torch.load(precalculated_data) 
            Tmat = sparse_mx_to_torch_sparse_tensor(Tmat)
            edge_adj = sparse_mx_to_torch_sparse_tensor(edge_adj)
            adj = sparse_mx_to_torch_sparse_tensor(adj)
        else:
            # we need to load and preprocessing the data from scratch 
            if data_name == 'cora': 
                idx_features_labels = np.genfromtxt("{}{}.content".format(path, data_name),
                                                    dtype=np.dtype(str))
                features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
                labels = encode_onehot(idx_features_labels[:, -1])
                print("after one hot operation")
                print(labels)
                # build graph
                idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
                idx_map = {j: i for i, j in enumerate(idx)}
                edges_unordered = np.genfromtxt("{}{}.cites".format(path, data_name),
                                                dtype=np.int32)
                edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                                 dtype=np.int32).reshape(edges_unordered.shape)
                adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                    shape=(labels.shape[0], labels.shape[0]),
                                    dtype=np.float32)
                print("diag")
                print(np.diag(adj.toarray()))
                print(np.where(np.diag(adj.toarray()) == 1))
                directed_adj = adj.toarray()
           
            elif data_name == 'citeseer':
                '''
                there are 15 papers in cite file that not appear in the content file
                so we remove those papers
                ['197556', '293457', '38137', '95786', 'flach99database',
                   'gabbard97taxonomy', 'ghani01hypertext', 'hahn98ontology',
                   'khardon99relational', 'kohrs99using', 'nielsen00designing',
                   'raisamo99evaluating', 'tobies99pspace', 'wang01process',
                   'weng95shoslifn']
                 '''   
                idx_features_labels = np.genfromtxt("{}{}.content".format(path, data_name),
                                                    dtype=np.dtype(str))
                features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
                labels = encode_onehot(idx_features_labels[:, -1])
                # build graph
                idx = np.array(idx_features_labels[:, 0], dtype=np.str)
                idx_map = {j: i for i, j in enumerate(idx)}

                edges_unordered = np.genfromtxt("{}{}.cites".format(path, data_name),
                                                dtype=np.str)
                # get the missing edges
                missing = np.setdiff1d(np.unique(edges_unordered.flatten()), idx)
                edges_unordered_new = np.array([[x, y] for x, y in edges_unordered if (x not in missing) 
                                                and (y not in missing)
                                                and (x != y)])

                edges = np.array(list(map(idx_map.get, edges_unordered_new.flatten())),
                                 dtype=np.str).reshape(edges_unordered_new.shape)
                adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                    shape=(labels.shape[0], labels.shape[0]),
                                    dtype=np.float32)
                directed_adj = adj.toarray() 


            elif data_name == 'pubmed':
                n_nodes = 19717
                n_features = 500
                n_classes = 3

                data_X = np.zeros((n_nodes, n_features), dtype='float32')
                data_Y = np.zeros((n_nodes, n_classes), dtype='int32')

                paper_to_index = {}
                feature_to_index = {}

                # parse nodes
                with open(path + 'Pubmed-Diabetes.NODE.paper.tab','r') as node_file:
                    # first two lines are headers
                    node_file.readline()
                    node_file.readline()
                    k = 0
                    for i,line in enumerate(node_file.readlines()):
                        items = line.strip().split('\t')

                        paper_id = items[0]
                        paper_to_index[paper_id] = i

                        # label=[1,2,3]
                        label = int(items[1].split('=')[-1]) - 1  # subtract 1 to zero-count
                        data_Y[i,label] = 1.

                        # f1=val1 \t f2=val2 \t ... \t fn=valn summary=...
                        features = items[2:-1]
                        for feature in features:
                            parts = feature.split('=')
                            fname = parts[0]
                            fvalue = float(parts[1])

                            if fname not in feature_to_index:
                                feature_to_index[fname] = k
                                k += 1

                            data_X[i, feature_to_index[fname]] = fvalue

                # parse graph
                data_A = np.zeros((n_nodes, n_nodes), dtype='float32')
                with open(path + 'Pubmed-Diabetes.DIRECTED.cites.tab','r') as edge_file:
                    # first two lines are headers
                    edge_file.readline()
                    edge_file.readline()

                    for i,line in enumerate(edge_file.readlines()):

                        # edge_id \t paper:tail \t | \t paper:head
                        items = line.strip().split('\t')

                        edge_id = items[0]

                        tail = items[1].split(':')[-1]
                        head = items[3].split(':')[-1]

                        data_A[paper_to_index[tail],paper_to_index[head]] = 1.0

                adj = sp.csr_matrix(data_A)
                directed_adj = adj
                features = sp.csr_matrix(data_X)
                labels = data_Y 
                
            # build symmetric adjacency matrix
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
            edge_adj, edge_name = create_edge_adj_new(adj.toarray())
            # create transition matrix
            Tmat = create_transition_matrix_new(adj.toarray())

            features = normalize(features)
            edge_features1 = node_corr_cosine(adj, features.todense())
            edge_features2 = create_direction_feature(directed_adj, edge_name)
            edge_features = np.concatenate([edge_features1, edge_features2], axis=1)
            adj = normalize(adj + sp.eye(adj.shape[0]))
            edge_adj = normalize(edge_adj + sp.eye(edge_adj.shape[0]))
            labels = torch.LongTensor(np.where(labels)[1])
            # deg = adj.todense().sum(axis=0).tolist()[0]

            features = torch.FloatTensor(np.array(features.todense()))
            
            edge_features = normalize(edge_features)
            edge_features = torch.FloatTensor(edge_features)
            edge_feature_dict = {}
            for i in range(len(edge_name)):
                edge_feature_dict[edge_name[i]] = edge_features[i, :]

            print("This is the first preprocessing data, so we save it for re-use")
            to_save_list = [Tmat, edge_adj, edge_name, edge_feature_dict, adj, features, edge_features, labels]
            torch.save(to_save_list, precalculated_data)
            
            # after saving to pt file, we should convert to pytorch sparse tensor
            Tmat = sparse_mx_to_torch_sparse_tensor(Tmat)
            edge_adj = sparse_mx_to_torch_sparse_tensor(edge_adj)
            adj = sparse_mx_to_torch_sparse_tensor(adj)

        # we randomly sampling certain ratio (say, x) for training, 0.5-x for val, and 0.5 for testing
        num_of_class=len(np.unique(labels))
        node_in_each_class= round((train_ratio * adj.shape[0])/num_of_class)
        idx_train = np.concatenate([np.random.choice(np.where(labels == i)[0], node_in_each_class, replace=False) for i in range(num_of_class)], axis=0)
        print("the number of train set is : ", len(idx_train))
        '''
        # let's match the number of nodes in each class
        print("importance sampling!")
        freqtable=np.array(np.unique(labels, return_counts=True)).T
        idx_train = []
        for i in range(freqtable.shape[0]):
            tmp_num = round(freqtable[i, 1].astype('float') * train_ratio).astype('int')
            tmp_idx = np.random.choice(np.where(labels == i)[0], tmp_num, replace=False)
            idx_train.append(tmp_idx)
        idx_train =np.concatenate(idx_train, axis=0)
        '''
        all_index = list(range(len(labels)))
        all_index_left = [x for x in all_index if x not in idx_train]
        np.random.shuffle(all_index_left)
        num_of_val = round((0.5 - train_ratio) * len(labels))
        idx_val = all_index_left[0:num_of_val]
        idx_test = all_index_left[num_of_val:len(all_index_left)] 
    
        # print seed
        print("======= PRINT HEAD OF IDS TO VERIFY======")
        print(idx_train[1:10], idx_test[1:10], idx_val[1:10])
        
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        return Tmat, edge_adj, edge_name, edge_feature_dict, adj, features, edge_features, labels, idx_train, idx_val, idx_test


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation
       If there is only 1 feature, then simply return it
    """
    if features.shape[0] == 1:
        return sparse_to_tuple(features)
    else:
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return (t_k)


def create_edge_adj(vertex_adj):
    '''
    create an edge adjacency matrix from vertex adjacency matrix
    '''
    vertex_adj.setdiag(0)
    edge_index = np.nonzero(sp.triu(vertex_adj, k=1))
    num_edge = int(len(edge_index[0]))
    edge_name = [x for x in zip(edge_index[0], edge_index[1])]

    edge_adj = np.zeros((num_edge, num_edge))
    for i in range(num_edge):
        for j in range(i, num_edge):
            if len(set(edge_name[i]) & set(edge_name[j])) == 0:
                edge_adj[i, j] = 0
            else:
                edge_adj[i, j] = 1
    adj = edge_adj + edge_adj.T
    np.fill_diagonal(adj, 1)
    return sp.csr_matrix(adj), edge_name


def create_transition_matrix(vertex_adj):
    '''create N_v * N_e transition matrix'''
    vertex_adj.setdiag(0)
    edge_index = np.nonzero(sp.triu(vertex_adj, k=1))
    num_edge = int(len(edge_index[0]))
    edge_name = [x for x in zip(edge_index[0], edge_index[1])]

    row_index = [i for sub in edge_name for i in sub]
    col_index = np.repeat([i for i in range(num_edge)], 2)

    data = np.ones(num_edge * 2)
    T = sp.csr_matrix((data, (row_index, col_index)),
               shape=(vertex_adj.shape[0], num_edge))

    return T


def create_edge_corr_feature(adj, feat):
    """calculate edge correlation strength"""
    prod = np.dot(feat, feat.T)
    edge_feat = prod[np.nonzero(sp.triu(adj, k=1))].T
    return sp.csr_matrix(edge_feat)


def create_edge_random_feature(num_edges):
    """calculate edge correlation strength"""
    edge_feat = np.random.randint(3, size=(num_edges, 5))
    return sp.csr_matrix(edge_feat)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1)).astype("float")
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
