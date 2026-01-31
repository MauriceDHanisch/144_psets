# %% [markdown]
# # Approximate betweenness centrality using neural networks
# Here we start to approximate the betweennesss centrality using neural networks over a peer-2-peer network Gnutella. Gnutella is a set of datasets consisting of 9 networks ranging from 6,300 to 63,000 nodes. Our goal is to train a neural network on the smallest Gnurella graph and evaluate it on a much larger graph. We will guide you through this step by step.
# 
# You can find Gnutella datasets at http://snap.stanford.edu/data/index.html. We will use p2p-Gnutella08 for training and p2p-Gnutella04 for testing.
# 
# Note:
# 1. Copy this notebook to your Google drive in order to execute it.
# 2. Make sure to upload the data files in HW4 to your google drive and to modify their corresponding directories in the code.

# %% [markdown]
# # Part 1: Training a model on Gnutella 08

# %% [markdown]
# ## Preprocessing Gnutella08 dataset
# 

# %%
import tensorflow as tf
import pandas as pd
import numpy as np
import networkx as nx
import scipy

# %%
# Parameters

# choose an embedding size for Structure2Vec
EMBED_SIZE = 64

# choose number of dense layers in the neural network
NUM_LAYERS = 5

# choose number of folds for cross validation
NUM_FOLD = 5

# choose number of epochs for training
NUM_EPOCHS = 20

# %%
# Normalize a list of values
# NO NEED TO CHANGE

def _normalize_array_by_rank(true_value, nr_nodes):
  # true_value is a list of values you want to normalize and nr_nodes is the number of nodes in the list

  rank = np.argsort(true_value, kind='mergesort', axis=None) #deg list get's normalised
  norm = np.empty([nr_nodes])

  for i in range(0, nr_nodes):

    norm[rank[i]] = float(i+1) / float(nr_nodes)

  max = np.amax(norm)
  min = np.amin(norm)
  if max > 0.0 and max > min:
    for i in range(0, nr_nodes):
      norm[i] = 2.0*(float(norm[i] - min) / float(max - min)) - 1.0
  else:
    print("Max value = 0")

  return norm, rank

# %%
#Read in and create NetworkX Graph; G

#TO-DO: The path needs to be changed according to your dataset directory in your GOOGLE DRIVE
path = '../data/p2p-Gnutella08.txt'

G = nx.read_edgelist(path, comments='#', delimiter=None, create_using=nx.DiGraph,
                  nodetype=None, data=True, edgetype=None, encoding='utf-8')

#print(nx.info(G))

# %%
# Creating list of Degrees of the nodes in G and normalising them:

deg_lst = [val for (node, val) in G.degree()]
nr_nodes = G.number_of_nodes()
print("deg_lst: \n", deg_lst)

degree_norm, degree_rank = _normalize_array_by_rank(deg_lst, nr_nodes)

# %%
# Computing Ground-truth values and normalising them:

b = [v for v in nx.betweenness_centrality(G).values()]

BC_norm_cent, BC_cent_rank = _normalize_array_by_rank(b, nr_nodes)

# Save the normalized betweenness centrality values
np.save("BC_norm_cent.npy", BC_norm_cent)

# Save teh cent rank
np.save("BC_cent_rank.npy", BC_cent_rank)

# %%
# Define Structure2Vec
# NO NEED TO CHANGE

def Structure2Vec(G, nr_nodes, degree_norm, num_features=1, embed_size=512, layers=2):

  #build feature matrix
  def get_degree(i):
    return degree_norm[i]

  def build_feature_matrix():
    n = nr_nodes
    feature_matrix = []
    for i in range(0, n):
      feature_matrix.append(get_degree(i))
    return feature_matrix
  #Structure2Vec node embedding
  A = nx.to_numpy_array(G)

  dim = [nr_nodes, num_features]


  node_features = tf.cast(build_feature_matrix(), tf.float32)
  node_features = tf.reshape(node_features, dim)

  initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0,
                                                                mode="fan_avg",
                                                                distribution="uniform")
  #print(initializer)

  A = tf.sparse.from_dense(A)
  A = tf.cast(A, tf.float32)
  w1 = tf.Variable(initializer((num_features, embed_size)), trainable=True,
                                  dtype=tf.float32, name="w1")
  w2 = tf.Variable(initializer((embed_size, embed_size)), trainable=True,
                                  dtype=tf.float32, name="w2")
  w3 = tf.Variable(initializer((1,embed_size)), trainable=True, dtype=tf.float32, name="w3")
  w4 = tf.Variable(initializer([]), trainable=True, dtype=tf.float32, name="w4")

  wx_all = tf.matmul(node_features, w1)  # NxE

  #computing X1:
  #sparse.reduce_sum: Computes the sum of elements across dimensions of a SparseTensor.
  weight_sum_init = tf.sparse.reduce_sum(A, axis=1, keepdims=True, ) #takes adjacency matrix
  n_nodes = tf.shape(input=A)[1]

  weight_sum = tf.multiply(weight_sum_init, w4)
  weight_sum = tf.nn.relu(weight_sum)  # Nx1
  weight_sum = tf.matmul(weight_sum, w3)  # NxE

  weight_wx = tf.add(wx_all, weight_sum)
  current_mu = tf.nn.relu(weight_wx)  # NxE = H^0

  for i in range(0, layers):
    neighbor_sum = tf.sparse.sparse_dense_matmul(A, current_mu)
    neighbor_linear = tf.matmul(neighbor_sum, w2)  # NxE

    current_mu = tf.nn.relu(tf.add(neighbor_linear, weight_wx))  # NxE

  mu_all = current_mu

  return mu_all

# %%
# Converting the graph structure into vectors

mu_all = Structure2Vec(G, nr_nodes, degree_norm, embed_size=EMBED_SIZE)

# %% [markdown]
# ## Training a Neural Network

# %%
# Building NN model

UNITS = int(EMBED_SIZE/2)
def build_model():
  model = tf.keras.Sequential()
  model.add(tf.keras.Input(shape=(EMBED_SIZE,)))

  # choose the number of layers to construct your network
  for _ in range(NUM_LAYERS):
    model.add(tf.keras.layers.Dense(UNITS, activation ="relu"))

  model.add(tf.keras.layers.Dense(1))
  model.compile(optimizer='sgd', loss='mse')

  model.summary()

  return model

model = build_model()

# %%
# Construct training set and groundtruth

x_train = mu_all
y_train = BC_norm_cent
print(tf.shape(x_train))
print(tf.shape(y_train))

# %%
# Computing cross validation
# NO NEED TO CHANGE
all_scores = []
k = NUM_FOLD
num_val_samples = len(x_train) // k
for i in range(k):
  print('processing fold #', i)
  val_data = x_train[i*num_val_samples: (i+1) * num_val_samples]
  val_targets = y_train[i*num_val_samples: (i+1)*num_val_samples]

  partial_train_data = np.concatenate(
      [x_train[:i*num_val_samples],
      x_train[(i+1)*num_val_samples:]],
      axis = 0)
  print(tf.shape(partial_train_data))

  partial_train_targets = np.concatenate(
      [y_train[:i*num_val_samples],
      y_train[(i+1)*num_val_samples:]],
      axis = 0)

  # Training
  callbacks =  tf.keras.callbacks.EarlyStopping(
      monitor= 'loss', min_delta=0, patience=3, verbose=1,
      mode='auto', baseline=None, restore_best_weights=False)

  model.fit(partial_train_data, partial_train_targets,
            epochs = NUM_EPOCHS, batch_size = 1, callbacks = callbacks, verbose = 1)
  print("model.metrics_names: ", model.metrics_names)

  val_loss = model.evaluate(val_data, val_targets, verbose = 1)

  all_scores.append(val_loss)
  print(all_scores)

# %%
# Computing Kendall on trained set

x_new = x_train
y_pred = model.predict(x_new)

# compute kendalltau using the prediction results and the groundtruth
from scipy import stats
kendall_tau, p_value = scipy.stats.kendalltau(BC_norm_cent,y_pred)

# %%
# Print your kendalltau score
# Make sure your kendalltau score is at least 0.70
# PRINT HERE
print(kendall_tau)

# %%
# You could save this model for part 2

model.save("../data/GN08_model_plain.h5")

# %% [markdown]
# # Part 2: Evaluating the trained model on Gnutella 04
# 
# Hints:
# 1. Write down the evaluation using the functions and codes in Part 1
# 2. Compute the groundtruth of betweenness centrality using NetworkX could take around 1 hour. Keep your Colab opened and be patient.

# %%
'''Gnutella 04'''
# change the path to your own directory
path2 = '../data/p2p-Gnutella04.txt'

G2 = nx.read_edgelist(path2, comments='#', delimiter=None, create_using=nx.DiGraph,
                  nodetype=None, data=True, edgetype=None, encoding='utf-8')

#print(nx.info(G2))


