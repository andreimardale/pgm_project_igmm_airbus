#ui
from IPython.display import display
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import numpy as np
import random, sys
from sklearn.manifold.t_sne import TSNE
from igmm import IGMM
import plotly.graph_objects as go

'''
Wrapper for the Normal Inverse Wishart prior distribution.
'''
class NIW(object):
    def __init__(self, m_0, k_0, v_0, S_0):
        self.m_0 = m_0
        self.k_0 = k_0
        self.v_0 = v_0
        self.S_0 = S_0


'''
User Interface class for tuning the values of the hyper-parameters and launching the execution.
'''
class IgmmUI(object):

  def __init__(self, data):
    self.data = data
    self.igmm = None
    self.record = None
    self.prior = None
    self.X_2d = None
    self.clusters_2d = None
    self.centers_2d = None
  
  def run_igmm(self, n_iter=5, alpha=5, k_0=5, v_0=1, S_0_scale=40):
    D = self.data.shape[1]

    S_0 = S_0_scale*np.ones(D)
    m_0 = np.zeros(D)

    self.prior = NIW(m_0, k_0, v_0, S_0*alpha)
    self.log_margs = np.zeros(n_iter)

    X = self.data
    self.igmm = IGMM(X, self.prior, alpha)
    self.record = self.igmm.gibbs_sample(n_iter)

    self.X_2d, self.clusters_2d, self.centers_2d = self.convert_2d()
    return True


  def tune_params(self):
    data = self.data
    D = data.shape[1]
    params = interactive(self.run_igmm, {'manual': True, 'manual_name': 'Run IGMM'}, n_iter=widgets.IntSlider(min=1,value=10, max=1000),
    alpha = widgets.FloatSlider(min=1, max=1000, value=5),
    k_0 = widgets.FloatSlider(min=0, max=300, value= 5),
    v_0 = widgets.IntSlider(min=D, max=D*3, value=D+3),
    S_0_scale = widgets.FloatSlider(min=0, max=300, value=1),
    X= data)
    return params

  def convert_2d(self):
    X = np.array(self.igmm.components.X)
    for k in range(self.igmm.components.K):
        mu,_ = self.igmm.components.rand_k(k)
        mu = mu.reshape(1,-1)
        X = np.concatenate((X, mu), axis=0)

    X_embedded_train_valid = TSNE(n_components=2).fit_transform(X) 
    y_train_valid = self.igmm.components.assignments

    centers = X_embedded_train_valid[self.igmm.components.X.shape[0]:]
    X_embedded_train_valid = X_embedded_train_valid[:self.igmm.components.X.shape[0]]

    return X_embedded_train_valid, y_train_valid, centers

  def visualize(self, X, clusters, centers):
    fig = go.Figure()
    for i in range(centers.shape[0]):
      fig.add_trace(go.Scatter(x = X[clusters == i,0],y = X[clusters == i,1], name='Cluster #' + str(i),marker=dict(size = 6, color = clusters[i]), mode="markers"))

    fig.add_trace(go.Scatter(x = centers[:,0],y = centers[:,1],name='Centers', marker=dict(symbol='star-dot', size = 10, color = 'red'), mode="markers"))

    fig.update_layout(title="", showlegend=True)
    fig.show()



