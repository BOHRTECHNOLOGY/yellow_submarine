import pandas as pd
from sklearn import tree
import graphviz
import random
import matplotlib.pyplot as plt
import itertools

full_data = pd.read_csv("results_2.csv")

data = full_data.sample(n=10000)
params = ['t','d','c','c_prim']
X = data[params]
Y = data.success
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("decisions") 

params_combinations = list(itertools.combinations(params, 2))

colors = data[['success']]
for p_1, p_2 in params_combinations:
    plt.scatter(data[[p_1]], data[[p_2]],c=colors,s=5)
    plt.xlabel(p_1)
    plt.ylabel(p_2)
    plt.show()