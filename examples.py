import sys

from scipy.io import loadmat as loadmat
from scipy.sparse import csr_matrix as csr_matrix

import amos

args = dict(map(lambda x: x.lstrip('-').split('='),sys.argv[1:]))
if('verbose' not in args):
    args['verbose'] = False;
if('Kmax' in args):
    Kmax = args['Kmax']
else:
    Kmax = 10 # maximum number of allowed clusters, smaller value of Kmax can speed up the computation process; otherwise Kmax can set to be n
if('k_init' in args):
    k_init = args['k_init']
else:
    k_init = 2

if 'n_jobs' in args:
    n_jobs = int(args['n_jobs'])
else:
    n_jobs = 1


if('A' in args):
    # Input data: A - the adjacency matrix of an undirected, weighted, and connected graph
    if(args['A'] == 'minnesota'):
        Kmax = 50
        A = loadmat('examples/Minnesota_road_adjacency_connected.mat')['A'] # Minnesota k is between 45 and 55
    elif(args['A'] == 'hibernia'):
        A =  csr_matrix(loadmat('examples/hibernia_adjacency.mat')['A'])  # hibertia k is 2
    elif(args['A'] == 'cogent'):
        A =  csr_matrix(loadmat('examples/cogent_adjacency.mat')['A'])  # cogent k is 4
    elif(args['A'] == 'ieee'):
        A =  loadmat('examples/IEEERTS96Adjacency.mat')['A']  # ieee k is 3
    elif(args['A'] == 'facebook'):
        A =  csr_matrix(loadmat('examples/facebook.mat')['A'])
    else:
        print 'No matrix A'
        print 'You can choose between minnesota, hibernia, cogent, ieee, facebook'

    model = amos.AMOS(k_init,Kmax,verbose=args['verbose'] == 'True',n_jobs=n_jobs)
    # best_k, labels = model.predict(A)
    model.predict(A)
    print 'number of found clusters is ',str(model.get_k())
else:
    print 'No matrix A selected'
    print 'You can choose between minnesota, hibernia, cogent, ieee, facebook'
