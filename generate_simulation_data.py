import networkx as nx
import numpy as np
import random
import math
n_feature = 1000
n_degree = 5
n_true = 20
n_negative= 5
sample_size = 500
def sigmoid(x):
    return 1. / (1. + np.exp(-x))
def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))




for it in range(1,2):
        print("##############################")
        print('it:', it)
        print("##############################")
        G=nx.random_graphs.barabasi_albert_graph(n_feature,n_degree)
        length = dict(nx.all_pairs_shortest_path_length(G))
        
        D = np.array([[length.get(m, {}).get(n, 0)  for m in G.nodes()] for n in G.nodes()], dtype=np.int32)        
        E = pow(0.7,D)       
        X = np.random.multivariate_normal(mean=np.repeat(1, n_feature), cov=E, size = sample_size) 

        # import ipdb; ipdb.set_trace()
        
        degree = np.array([dict(G.degree()).get(m,{}) for m in G.nodes()],dtype=np.int32) #
        positive = np.argsort(-degree)[:n_true]
        X_ture =X[:,positive]
        b = np.random.uniform(1,1.5,n_true+1) #
        negative_id = random.sample(range(n_true), n_negative) 
        b[negative_id] = -b[negative_id]
        score_matrix = b[:n_true]* X_ture + b[n_true:]
        score_pre = np.sum(score_matrix,axis=1)
        score_1 = sigmoid(score_pre) #sigmoid function
        score_2 = 0.7*(
            tanh(score_pre)
            -min(tanh(score_pre))
        )/(
            max(tanh(score_pre))-min(tanh(score_pre))
        )+0.3*(np.square(score_pre)-min(np.square(score_pre)))/(max(np.square(score_pre))-min(np.square(score_pre)))#
        # import ipdb; ipdb.set_trace()
        score=score_2/score_1*score_1 
        # score=score_1
        y =np.repeat(0,sample_size) #
        y[score > np.median(score)] = 1#
        # y = score
        A = E #
        A[A>0.5] = 1
        A[A<0.5] = 0
        

        np.savetxt("./simulation/generate_gedfn/gedfn%d_x.txt" % it,X)
        np.savetxt("./simulation/generate_gedfn/gedfn%d_position.txt" % it,positive)
        np.savetxt("./simulation/generate_gedfn/gedfn%d_gene_A.txt"% it,A)
        np.savetxt("./simulation/generate_gedfn/gedfn%d_y.txt"% it,y)

        print("END")
