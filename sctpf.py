import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MSE, KLD
from tensorflow.keras.layers import Dense, Dropout, Input, Lambda
from spektral.layers import TAGConv
from tensorflow.keras.initializers import GlorotUniform
from layers import *
import tensorflow_probability as tfp
import tensorflow as tf
########################################################################
from knn_utils import *
from scipy.stats import mode
from sklearn.cluster import KMeans
########################################################################
from sklearn import metrics
from loss import ZINB, dist_loss
from sklearn.neighbors import NearestNeighbors
from graph_function import *
import os
import csv

def linear_assign(y_true,y_pred):
    '''
    Returns a linear assignment between two lists of labels
    '''
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return ind

def cluster_acc(y_true, y_pred, ind):
    '''
    Returns the clustering accuracy given true and predicted labels as well as a linear assignment
    '''
    true_labeled = 0
    y_true_cp = y_true.copy()
    y_pred_cp = y_pred.copy()
    if len(y_pred) > 0:
        for i in range(len(y_pred_cp)):
            y_pred_cp[i] = ind[y_pred_cp[i],1]
            if y_pred_cp[i] == y_true_cp[i]:
                true_labeled+=1

        return true_labeled * 1.0/ y_pred_cp.size
    else:
        return 1

MeanAct = lambda x: tf.clip_by_value(K.exp(x), 1e-5, 1e6)
DispAct = lambda x: tf.clip_by_value(tf.nn.softplus(x), 1e-4, 1e4)


def thresholding(p, beta_1=0, beta_2=0.9):
    '''
    Thresholding filter to identify datapoints with high-confidence clustering assignment
    beta_2: the threshold over which the soft assignment is considered  reliable(epsilon in Algorithm 1 of the paper)
    '''
    unconf_indices = []
    conf_indices = []
    p = p.numpy()
    confidence1 = p.max(1)
    confidence2 = np.zeros((p.shape[0],))
    a = np.argsort(p, axis=1)[:, -2]
    for i in range(p.shape[0]):
        confidence2[i] = p[i, a[i]]
        if (confidence1[i] > beta_1) and (confidence1[i] - confidence2[i]) > beta_2:
            unconf_indices.append(i)
        else:
            conf_indices.append(i)
    unconf_indices = np.asarray(unconf_indices, dtype=int)
    conf_indices = np.asarray(conf_indices, dtype=int)
    return unconf_indices, conf_indices


def generate_unconflicted_data_index(q,ntrain,labaels_matrix,features,pred_labels,nclass,adj,k_cc=2,cp_opt=3,zeta=0.5,k_outlier=60,beta_2=0.9):
    '''
    This function returns datapoints with high-confidence clustering assignment based on a topological and a probabilistic filter
    beta_2:the threshold over which soft assignment is considered  reliable(epsilon Algorithm 1 of the paper)
    k_cc: the parameter used to build the k-nn graph(k in  Algorithm 1 of the paper )
    '''
    unconf, conf = thresholding(q,beta_2=beta_2)
    conf = set(conf)
    ntrain = len(unconf)
    labaels_matrix = labaels_matrix[unconf]
    features = features[unconf]
    pred_labels = pred_labels[unconf]
    #Computing the largest components of the k_cc-nn graph
    _, idx_of_comp_idx2 = calc_topo_weights_with_components_idx(ntrain, labaels_matrix, features,
                                                                pred_labels, pred_labels, adj=adj[unconf][:,unconf], k=k_cc,
                                                                cp_opt=cp_opt, nclass=nclass)
    C = set(range(ntrain)) - set(idx_of_comp_idx2)
    C_idx = list(C)
    feats_C = features[C_idx]
    labels_C = np.array(pred_labels)[C_idx]
    knnG_list = calc_knn_graph(feats_C, k=k_outlier)
    knnG_list = np.array(knnG_list)
    knnG_shape = knnG_list.shape
    knn_labels = labels_C[knnG_list.ravel()]
    knn_labels = np.reshape(knn_labels, knnG_shape)

    majority, counts = mode(knn_labels, axis=-1)
    majority = majority.ravel()
    counts = counts.ravel()

    #zeta filtering
    non_outlier_idx = np.where((majority == labels_C) & (counts >= k_outlier * zeta))[0]
    outlier_idx = np.where(majority != labels_C)[0]
    outlier_idx = np.array(list(C))[outlier_idx]

    C = np.array(list(C))[non_outlier_idx]
    C = set(C.tolist())
    idx_added_conf = list(set(np.arange(ntrain).tolist())-C)
    added_conf = np.array(unconf)[idx_added_conf].tolist()
    C = np.array(unconf)[list(C)].tolist()
    unconflicted = np.array(C)
    conflicted = np.array(list(conf.union(set(added_conf))))

    return unconflicted,conflicted,added_conf

class SCTPF(tf.keras.Model):

    def __init__(self, X, adj, adj_n,y, hidden_dim=128, latent_dim=15, dec_dim=None, adj_dim=32):
        super(SCTPF, self).__init__()
        if dec_dim is None:
            dec_dim = [128, 256, 512]
            #dec_dim = [128, 256]
        self.latent_dim = latent_dim
        self.X = X
        self.adj = np.float32(adj)
        self.adj_n = np.float32(adj_n)
        self.y = y
        self.n_sample = X.shape[0]
        self.in_dim = X.shape[1]
        self.sparse = False

        initializer = GlorotUniform(seed=7)
        '''
        The architecture of the graph autoencoder. This piece of code is similar to scTAG(https://github.com/Philyzh8/scTAG)  
        '''
        # The graph convolutionl encoder
        X_input = Input(shape=self.in_dim)
        h = Dropout(0.2)(X_input)
        self.sparse = True
        A_in = Input(shape=self.n_sample, sparse=True)
        h = TAGConv(channels=hidden_dim, kernel_initializer=initializer, activation="relu")([h, A_in])
        z_mean = TAGConv(channels=latent_dim, kernel_initializer=initializer)([h, A_in])
        self.encoder = Model(inputs=[X_input, A_in], outputs=z_mean, name="encoder")
        clustering_layer = ClusteringLayer(name='clustering')(z_mean)
        self.cluster_model = Model(inputs=[X_input, A_in], outputs=clustering_layer, name="cluster_encoder")

        # Adjacency matrix decoder
        dec_in = Input(shape=latent_dim)
        h = Dense(units=adj_dim, activation=None)(dec_in)
        h = Bilinear()(h)
        dec_out = Lambda(lambda z: tf.nn.sigmoid(z))(h)
        self.decoderA = Model(inputs=dec_in, outputs=dec_out, name="decoder1")

        #The gene count matrix decoder based on the Zero-Inflated Negative Binomial Model (ZINB)
        decx_in = Input(shape=latent_dim)
        h = Dense(units=dec_dim[0], activation="relu")(decx_in)
        h = Dense(units=dec_dim[1], activation="relu")(h)
        h = Dense(units=dec_dim[2], activation="relu")(h)

        pi = Dense(units=self.in_dim, activation='sigmoid', kernel_initializer='glorot_uniform', name='pi')(h)

        disp = Dense(units=self.in_dim, activation=DispAct, kernel_initializer='glorot_uniform', name='dispersion')(h)

        mean = Dense(units=self.in_dim, activation=MeanAct, kernel_initializer='glorot_uniform', name='mean')(h)

        self.decoderX = Model(inputs=decx_in, outputs=[pi, disp, mean], name="decoderX")

    def pre_train(self, epochs=1000, info_step=10, lr=1e-4, W_a=0.3, W_x=1, W_d=0, min_dist=0.5, max_dist=20):
        '''
        The pretraining step optimizes TWO loss functions: A graph reconstruction loss and a ZINB loss
        '''
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        if self.sparse == True:
            self.adj_n = tfp.math.dense_to_sparse(self.adj_n)

        # Training
        for epoch in range(1, epochs + 1):
            with tf.GradientTape(persistent=True) as tape:
                z = self.encoder([self.X, self.adj_n])
                # X_out = self.decoderX(z)
                pi, disp, mean = self.decoderX(z)
                A_out = self.decoderA(z)

                if W_d:
                    Dist_loss = tf.reduce_mean(dist_loss(z, min_dist, max_dist=max_dist))
                A_rec_loss = tf.reduce_mean(MSE(self.adj, A_out))
                zinb = ZINB(pi, theta=disp, ridge_lambda=0, debug=False)
                zinb_loss = zinb.loss(self.X, mean, mean=True)
                loss = W_a * A_rec_loss + W_x * zinb_loss
                if W_d:
                    loss += W_d * Dist_loss

            vars = self.trainable_weights
            grads = tape.gradient(loss, vars)
            optimizer.apply_gradients(zip(grads, vars))
            if epoch % info_step == 0:
                if W_d:
                    print("Epoch", epoch, " zinb_loss:", zinb_loss.numpy(), "  A_rec_loss:", A_rec_loss.numpy(),
                         "Dist_loss:", Dist_loss.numpy())

                else:
                    print("Epoch", epoch, " zinb_loss:", zinb_loss.numpy(), "  A_rec_loss:", A_rec_loss.numpy())

        print("Pre_train Finish!")

    def alt_train(self, y, epochs=300, lr=5e-4, W_a=0.3, W_x=1, W_c=1.5, info_step=8, n_update=8, centers=None, save_path='./saving_path', save = False, threshold_2=0.7,beta_1= 0.95,old=False,k_cc=2,zeta=0.5,k_outlier=60):
        '''
        The clustering step optimizes THREE loss functions: A graph reconstruction loss, a ZINB-based loss and KL-divergence loss
        '''
        self.cluster_model.get_layer(name='clustering').clusters = centers
        # Training

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr,beta_1=beta_1)
        best_acc = 0
        best_nmi = 0
        best_ari = 0
        if save == True:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            logfile = open(save_path + '/log_train.csv', 'w')

            logwriter = csv.DictWriter(logfile,fieldnames=['iter','acc', 'nmi', 'ari', 'unconf_acc', 'conf_acc',
                                                   'nb_unconf', 'nb_conf', 'nb_added_links',
                                                   'nb_false_added_links',
                                                   'nb_true_added_links', 'nb_dropped_links', 'nb_false_dropped_links',
                                                   'nb_true_dropped_links','nb_unconf','nb_conf'])

            logwriter.writeheader()
            count_target_links = {"nb_added_links": 0,
                                  "nb_false_added_links": 0,
                                  "nb_true_added_links": 0,
                                  "nb_deleted_links": 0,
                                  "nb_false_deleted_links": 0,
                                  "nb_true_deleted_links": 0}

        adj_norm_pos = self.adj_n
        adj_pos = self.adj

        for epoch in range(0, epochs):
            with tf.GradientTape(persistent=True) as tape:
                z = self.encoder([self.X, adj_norm_pos])
                q_out = self.cluster_model([self.X, adj_norm_pos])
                pi, disp, mean = self.decoderX(z)
                A_out = self.decoderA(z)


                if epoch % n_update == 0:
                    ntrain = self.adj.shape[0]
                    labels_matrix = np.zeros_like(q_out.numpy())
                    labels_matrix[np.arange(len(q_out)), q_out.numpy().argmax(1)] = 1.0
                    features = z.numpy()
                    pred_labels = q_out.numpy().argmax(1)
                    '''Compute unconflicted and conflicted lists'''
                    unconflicted_ind, conflicted_ind, added_conf_ind = generate_unconflicted_data_index(q_out,ntrain,labels_matrix,features,pred_labels,nclass=max(y)+1,adj = self.adj,k_cc=k_cc,zeta=zeta,k_outlier=k_outlier,beta_2=threshold_2)
                    print('Unconflicted: ', len(unconflicted_ind))
                    p = self.target_distribution(q_out, unconflicted_ind, conflicted_ind)
                    if old == False:
                        '''Update the input graph each n_update steps'''
                        adj_pos, adj_norm_pos = self.update_graph(unconflicted_ind)
                        adj_pos = adj_pos.toarray()
                        adj_norm_pos = tf.sparse.from_dense(adj_norm_pos.toarray())

                A_rec_loss = tf.reduce_mean(MSE(adj_pos, A_out))
                zinb = ZINB(pi, theta=disp, ridge_lambda=0, debug=False)
                zinb_loss = zinb.loss(self.X, mean, mean=True)
                cluster_loss = tf.reduce_mean(KLD(q_out, p))
                tot_loss = W_a * A_rec_loss + W_x * zinb_loss + W_c * cluster_loss

            vars = self.trainable_weights
            grads = tape.gradient(tot_loss, vars)
            optimizer.apply_gradients(zip(grads, vars))

            if epoch % info_step == 0:
                print("epoch: ", epoch)
                #pred = tf.math.argmax(q_out,1).numpy()
                pred = KMeans(n_clusters = len(set(self.y))).fit_predict(z)
                ind = linear_assign(y, pred)
                acc = np.round(cluster_acc(y, pred, ind), 5)
                acc_unconflicted = np.round(cluster_acc(y[unconflicted_ind], pred[unconflicted_ind], ind), 5)
                acc_conflicted = np.round(cluster_acc(y[conflicted_ind], pred[conflicted_ind], ind), 5)
                acc_added_conf = np.round(cluster_acc(y[added_conf_ind], pred[added_conf_ind], ind), 5)
                y = np.array(list(map(int, y)))
                nmi = np.round(metrics.normalized_mutual_info_score(y, pred), 5)
                ari = np.round(metrics.adjusted_rand_score(y, pred), 5)
                if save == True:
                    logdict = dict(iter=epoch,
                                   acc=acc, nmi=nmi, ari=ari,
                                   unconf_acc=acc_unconflicted, conf_acc=acc_conflicted,
                                   nb_added_links=count_target_links["nb_added_links"],
                                   nb_false_added_links=count_target_links["nb_false_added_links"],
                                   nb_true_added_links=count_target_links["nb_true_added_links"],
                                   nb_dropped_links=count_target_links["nb_deleted_links"],
                                   nb_false_dropped_links=count_target_links["nb_false_deleted_links"],
                                   nb_true_dropped_links=count_target_links["nb_true_deleted_links"],
                                   nb_unconf=len(unconflicted_ind), nb_conf=len(conflicted_ind))
                    logwriter.writerow(logdict)
                    logfile.flush()
                if acc>best_acc:
                    best_acc = acc
                    if nmi>best_nmi:
                        best_nmi = nmi
                    if ari>best_ari:
                        best_ari = ari
                    if save == True:
                        self.save_weights(save_path+'/best_model_weights.pkl')
                print("Epoch", epoch, " zinb_loss: ", zinb_loss.numpy(), " A_rec_loss: ", A_rec_loss.numpy(),
                      " cluster_loss: ", cluster_loss.numpy(), " acc: ",acc,"nmi: ",nmi,"ari: ",ari, " acc_unconflicted: ",acc_unconflicted, " acc_conflicted: ",acc_conflicted,"acc_added_conf:",acc_added_conf)

        tf.compat.v1.disable_eager_execution()
        q = tf.constant(q_out)
        session = tf.compat.v1.Session()
        q = session.run(q)
        self.y_pred = q.argmax(1)
        print('best_acc: ', best_acc)
        print('best_nmi: ', best_nmi)
        print('best_ari: ', best_ari)
        return self

    def target_distribution(self, p,unconflicted_ind, conflicted_ind):
        '''
        q[i] = p[i] if i belongs to the conflicted indexes
        q[i,argmax(p[i])] = 1 and q[i,j≠argmax(p[i])] = 0 if i belongs to the belongs to the unconflicted indexes
        '''
        p = p.numpy()
        q = np.zeros(p.shape)
        q[conflicted_ind] = p[conflicted_ind]
        q[unconflicted_ind, np.argmax(p[unconflicted_ind], axis=1)] = 1
        q = tf.convert_to_tensor(q,dtype=tf.float32)
        return q

    def embedding(self, count, adj_n):
        if self.sparse and not isinstance(adj_n,tf.sparse.SparseTensor):
            adj_n = tfp.math.dense_to_sparse(adj_n)
        return np.array(self.encoder([count, adj_n]))

    def rec_A(self, count, adj_n):
        h = self.encoder([count, adj_n])
        rec_A = self.decoderA(h)
        return np.array(rec_A)

    def get_label(self, count, adj_n):
        if self.sparse and not isinstance(adj_n,tf.sparse.SparseTensor):
            adj_n = tfp.math.dense_to_sparse(adj_n)
        clusters = self.cluster_model([count, adj_n]).numpy()
        labels = np.array(clusters.argmax(1))
        return labels.reshape(-1, )

    def update_graph(self, unconf_indices):
        '''
        Update the input graph in order to make it clustering relevant
        '''
        y_pred = self.get_label(self.X,self.adj_n)
        adj_pos = sp.lil_matrix(self.adj)
        idx = unconf_indices[self.generate_centers(unconf_indices)]
        for i, k in enumerate(unconf_indices):
         adj_k_pos = adj_pos[k].tocsr().indices
         if not(np.isin(idx[i], adj_k_pos)) and (y_pred[k] == y_pred[idx[i]]):
             adj_pos[k, idx[i]] = 1
        adj_pos = adj_pos - sp.dia_matrix((adj_pos.diagonal()[np.newaxis, :], [0]), shape=adj_pos.shape)
        adj_pos = adj_pos.tocsr()
        adj_pos.eliminate_zeros()
        adj_norm_pos = norm_adj(adj_pos)
        return adj_pos, adj_norm_pos


    def generate_centers(self, unconf_indices):
        y_pred = self.get_label(self.X, self.adj_n)[unconf_indices]
        emb_unconf = self.embedding(self.X, self.adj_n)[unconf_indices]
        nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(emb_unconf)
        _, indices = nn.kneighbors(self.cluster_model.get_layer(name='clustering').clusters)
        return indices[y_pred]
