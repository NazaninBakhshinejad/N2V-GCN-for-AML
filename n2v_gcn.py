import pandas as pd
import numpy as np
import networkx as nx
from node2vec import Node2Vec
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from tensorflow.keras.metrics import Metric
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from spektral.layers import GCNConv


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def split_data(df):
    X = df.drop(['isFraud'], axis=1)
    y = df['isFraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test


def oversample_data(X_train, y_train):
    sm = SMOTE(random_state=42, sampling_strategy=1)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    return X_train, y_train


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def convert_to_graph(df):
    G = nx.from_pandas_edgelist(df, source=source_customer, target=destination_customer, edge_attr=True)
    return G


def generate_embeddings(G, dimensions=128, walk_length=32, num_walks=15):
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    account_embeddings = {}
    for node in G.nodes():
        account_embeddings[node] = model.wv[node]
    return G, account_embeddings


def generate_feature_vectors(G, X, account_embeddings):
    X_embedded = np.zeros((len(X), 256))
    for i in range(len(X)):
        src = X[i][0]
        dst = X[i][1]
        X_embedded[i] = np.concatenate((account_embeddings[src], account_embeddings[dst]))
    return G, X_embedded

class FalseNegativeRate(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fn = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)
        false_negatives = tf.cast(tf.logical_and(y_true, tf.logical_not(y_pred)), self.dtype)
        self.fn.assign_add(tf.reduce_sum(false_negatives))

    def result(self):
        return self.fn / (self.fn + self.tp + self.fp)


def create_model(G, X_embedded):
    node_features = np.array([account_embeddings[str(node)] for node in G.nodes()])
    edge_indices = np.array(G.edges())
    edges = tf.SparseTensor(indices=edge_indices, values=np.ones(len(edge_indices)), dense_shape=[len(G.nodes()), len(G.nodes())])
    inputs = tf.keras.layers.SparseTensorValue(indices=edges.indices, values=edges.values, dense_shape=edges.dense_shape)
    X_input = tf.keras.layers.Input(tensor=inputs, shape=(None,))
    X_node = tf.keras.layers.Input(shape=(256,))
    conv = GCNConv(64, activation='relu')([X_input, X_node])
    conv = Dropout(0.5)(conv)
    conv = GCNConv(32, activation='relu')(conv)
    conv = Dropout(0.5)(conv)
    output = Dense(1, activation='sigmoid')(conv)
    model = Model(inputs=[X_input, X_node], outputs=output)
    #model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    #model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy', tf.keras.metrics.FalseNegatives()])
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=[FalseNegativeRate()])
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs=15, batch_size=32):
    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    model = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[early_stop])
    return model

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    return recall, precision, fpr, fnr

def predict(model, X):
    y_pred = model.predict(X)
    y_pred_classes = np.argmax(y_pred, axis=1)
    return y_pred_classes
