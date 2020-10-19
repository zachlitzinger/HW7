import numpy as np
from numpy import random
import matplotlib.pyplot as plt

def power_method(a, steps, solution):

    n = a.shape[0]
    x = random.rand(n)
    it = 0
    error = []


    while it < steps: 
        q = np.dot(a, x)  
        x = q/np.sqrt(q.dot(q))
        err = abs(x-solution)
        error.append(err)
        it += 1
     
    return error


def stationary(pt, steps):

    x = random.rand(pt.shape[0])  
    x /= sum(x)
    x1 = np.dot(pt, x)
    x1 /= sum(x1)
    error = 10
    it = 0 
    while it < steps and error > .0001:
        x[:] = x1
        x1 = np.dot(pt,x)
        x1 /= sum(x1)
        error = np.max(np.abs(x-x1))
        it += 1

    return x1

def pagerank_small():
    n = 5
    p_matrix = np.array([[0, 1/3, 1/3, 1/3, 0],
                         [1/2, 0, 0, 0, 1/2],
                         [1/2, 1/2, 0, 0, 0],
                         [1/2, 1/2, 0, 0, 0],
                         [0, 1, 0, 0, 0]])

    alpha = 0.95
    pt_mod = np.zeros((n, n))
    for j in range(n):
        for k in range(n):
            pt_mod[j, k] = alpha*p_matrix[k, j] + (1-alpha)*(1/n)

    dist = stationary(pt_mod,100)
    print(dist)


def read_graph_file(fname):
    adjList = {}
    names = {}

    with open(fname, 'r') as fp:
        for line in fp:
            separate = line.split(' ')
            if len(separate) < 4:
                continue
            node = int(separate[1])
            if separate and separate[0][0] == 'n':
                names[node] = separate[2].strip('\n')
            elif separate and separate[0][0] == 'e':
                vertex = int(separate[2])
                if node not in adjList:
                    adjList[node] = [vertex]
                else:
                    adjList[node].append(vertex)

    return adjList, names
    



if __name__ == "__main__":
    a = np.array([[3, 1], [0, 2]])

    solution = [1, 0]

    error = power_method(a, 100, solution)
    
    x = np.linspace(0, 100)
    ref = .75**x
   
    plt.semilogy(error)
    plt.semilogy(ref)
    plt.ylabel("error")
    plt.xlabel("n")
    plt.savefig('q1.pdf')
    plt.show()

    pagerank_small()
    #question 2b and 2c
    # this returns [0.27092134 0.35763124 0.09576501 0.09576501 0.17991741], and takes about 15 iterations to converge
    # and there is no aplha value that will change the highest ranked page after decreasing the value.