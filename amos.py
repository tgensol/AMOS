""" Automated model order selection (AMOS) algorithm for spectral graph clustering (SGC)
"""

# Author: Pin-Yu Chen <pinyu@umich.edu>, Thibaut Gensollen <thibaut.gensollen@ens-cachan.fr>, Alfred O. Hero III, Fellow, IEEE <hero@umich.edu>
# License: BSD 3 clause
# Copyright: University of Michigan


from __future__ import division
import warnings
from math import asin
from sympy import binomial
import numpy as np

from scipy.sparse.linalg import norm as normSparse
from scipy.sparse.linalg import eigsh
from scipy.sparse import eye, spdiags, issparse, csr_matrix
from scipy.sparse.csgraph import connected_components

from scipy.stats import chi2, norm
from sklearn.cluster import KMeans as skKmeans

from joblib import Parallel,delayed

def controlIdxLen(idx):
    """ Scipy has an issue if the array idx is only with a length of one
    """
    if idx.size == 1:
        idx = idx[0]
    return idx


###################### AMOS ######################

def spectral_bound_est_weighted(Acc,labels,nest,k,alge_trace,verbose=False):
    """ Phase transition bounds estimation
    """
    if(len(alge_trace) != k):
        # error
        raise Exception('error', 'len(alge_trace) != k')
    nmax = np.max(nest)
    nmin = np.min(nest)
    alge_trace_min = np.min(alge_trace)
    
    num_interedge=0;
    inter_edge_weight=0;
    West = np.zeros((k, k))
    west = 0
    for i in range(0,k):
        idx = np.nonzero(i==labels)[0]
        idx = idx.reshape(idx.shape+(1,))
        for j in range(i+1,k):
            idx2 = np.nonzero(j==labels)[0]
            idx2 = idx2.reshape(idx2.shape+(1,))
            # scipy has an issue with single array value
            idx2 = controlIdxLen(idx2)
            tmpW= Acc[idx,np.transpose(idx2)]
            temp = tmpW.nonzero()
            nnz = tmpW[temp]
            lenZero = len(temp[0])
            if(lenZero == 0):
                West[i][j] = 0;
            else:
                West[i][j] = np.mean(nnz)
                num_interedge += lenZero
                inter_edge_weight += nnz.sum()

    if(num_interedge): west = inter_edge_weight / num_interedge
                
    tmp = alge_trace_min/(k-1)
    tLB = tmp / nmax;
    tUB = tmp / nmin;
    
    return tLB, tUB, West, west

def confidence_interval_spec_new(Aconn,k,labels,nest,alpha,verbose=False):
    """  Homogeneous RIM test 
    """
    sym = int(binomial(k, 2))
    # Chi-square inverse cumulative distribution function :
    upp_quan = chi2.ppf(1-alpha/2, (sym-1),k/2)
    low_quan = chi2.ppf(alpha/2, (sym-1),k/2)

    # MLE estimate
    n = len(labels);
    m = len(Aconn.nonzero()[0])/2;
    mest = np.zeros((k,k));
    Pest = np.zeros((k,k));
    for i in range(0,k):
        idx = np.nonzero(i==labels)[0]
        idx = idx.reshape(idx.shape+(1,))
        idx = controlIdxLen(idx)
        for j in range(i,k):
            idx2 = np.nonzero(j==labels)[0]
            idx2 = idx2.reshape(idx2.shape+(1,))
            idx2 = np.transpose(controlIdxLen(idx2))
            if(i == j):
                mest[i][j]= len(Aconn[idx,idx2].nonzero()[0])/2
            else:
                mest[i][j]= len(Aconn[idx,idx2].nonzero()[0])

    # Pest is triangle
    for i in range(0,k):
        for j in range(i+1,k):
            if((np.dot(nest[i],nest[j])) == 0):
                if(verbose): print 'zero cluster size'
            else:
                Pest[i,j] = mest[i][j]/np.dot(nest[i],nest[j])
    pest=2*(m-mest.trace())/(np.dot(n,n)-np.sum(np.square(nest)))
    
    LB_approx=0; UB_approx=0;

    PestTmp = Pest;

    PestTmp[PestTmp == 0]=1;
    
    if(k>=3):
        log1 = np.log(PestTmp)
        log1[np.isneginf(log1)] = 0
        
        log2 = np.log(1-PestTmp)
        log2[np.isneginf(log2)] = 0
        
        log3 = np.log(pest)
        if(log3 == float('Inf')):
            log3 = 0
        log4 = np.log(1-pest)
        if(log4 == float('Inf')):
            log4 = 0
            
        tmp = 2*np.multiply(mest,log1).sum() + 2*np.multiply(np.add(np.outer(nest,nest),-mest),log2).sum()
        tmp += -2*(m-mest.trace())*log3
        tmp += -(n*n-sum(np.square(nest))-2*(m-mest.trace()))*log4

        if((tmp<=upp_quan)and(tmp>=low_quan)):
            CI_ind=1;
        else:
            CI_ind=0;
    else:
        CI_ind=1;    
    return CI_ind,mest,pest,Pest

def estimate_Inhom_Weight(i,k,labels,Pest,tLB,West):
    """ Inhomogeneous RIM phase transition test 
    """
    c=3/8;
    tmp = 1
    idx = np.nonzero(i==labels)[0]
    idx = idx.reshape(idx.shape+(1,))
    for j in range(i+1,k):
        idx2 = np.nonzero(j==labels)[0]
        idx2 = idx2.reshape(idx2.shape+(1,))
        n1 = len(idx)
        n2 = len(idx2)
        n = n1*n2;
        if(Pest[i][j] == 0):
            tmp = tmp*(0 < tLB)
        elif (Pest[i][j]==1):
            tmp = tmp*(West[i,j] < tLB)
        else:
            x1 = asin(np.sqrt(tLB/West[i,j]+(c/n)/(1+2*c/n)));
            x2 = asin(np.sqrt(Pest[i,j]+(c/n)/(1+2*c/n)));
            x = np.sqrt(4*n+2)*(x1-x2);
            tmp *= norm.cdf(x);
    return tmp

def confidence_interval_inhom_RIM_Anscombe_InhomWeight(beta,West,Pest,tLB,k,labels,n_jobs,verbose=False):
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(estimate_Inhom_Weight)(i,k,labels,Pest,tLB,West) for i in range(0,k))

    if(reduce(lambda a, b: a * b, results)>=1-beta):
        CI_ind=1;
    else:
        CI_ind=0;
    return CI_ind


def compute_alge_trace(idx,Acc,k,verbose=False):
    """  Compute partial eigenvalue sum
    """
    
    Atmp = Acc[idx,np.transpose(idx)]
    n = Atmp.shape[0]
    d = Atmp.sum(axis=1)
    D = spdiags(np.transpose(d).reshape(-1),0,n,n,format='csr')
    L = np.add(D,-Atmp)
    if(n<=1):
        if(verbose): print 'single or no cluster'
        return 0
    elif (n<=k):
        if(verbose): print 'small cluster'
        return Atmp.sum()
    else:
        return eigsh(L,k=k,which='SM',return_eigenvectors=False).sum()

def compute_alge_trace_parallel(labels,Acc,i,k):
    idx = np.nonzero(i==labels)[0]
    idx = idx.reshape(idx.shape+(1,))  
    idx = controlIdxLen(idx)
    return compute_alge_trace(idx,Acc,k), len(idx)

def rimTest(Aconn,labels,k,Acc,pth,n_jobs=1,verbose=False):
    """ V-test for testing the fitness of RIM, with p-value computation

    """

    flag_CI = True
    ci_condition = 0

    results = Parallel(n_jobs, verbose=verbose)(delayed(compute_alge_trace_parallel)(labels,Acc,i,k) for i in range(0,k))
    alge_trace, nest = zip(*results)

    # p-value
    # local homogeneity testing
    Pvalue = np.zeros((k, k))
    for i in range(0,k):
        idx = np.nonzero(i==labels)[0]
        idx = idx.reshape(idx.shape+(1,))
        idx = controlIdxLen(idx)
        for j in range(i+1,k):
            idx2 = np.nonzero(j==labels)[0]
            idx2 = idx2.reshape(idx2.shape+(1,))
            idx2 = controlIdxLen(idx2)
            tempC= Aconn[idx,idx2.T]
            zeros = tempC.nonzero()
            
            nnz = tempC[zeros]
            lenZero = len(zeros[0])
            if(lenZero == 0 or lenZero == nest[i]*nest[j]):
                Pvalue[i][j] = 1;
            else:
                n1 = len(idx)
                n2 = len(idx2)
                x = tempC.sum(axis=1)
                ide = np.full(tempC.shape, 1,np.double)
                y = (ide-tempC).sum(axis=1)
                X = np.dot(np.transpose(x),x)-x.sum()
                Y = np.dot(np.transpose(y),y)-y.sum()
               
                N = n1*n2*(n2-1)
                
                V = np.square(np.sqrt(X)+np.sqrt(Y))
                Z = (V-N)/np.sqrt(2*N)
                nrm = norm.cdf(Z)
                Pvalue[i][j] = 2*np.minimum(nrm,1-nrm)
                if(Pvalue[i][j] < pth):
                    # RIM Pvalue
                    if(verbose): print 'Reject RIM'
                    flag_CI = False;
                    break;
        else:
            continue
        break;
    if(flag_CI):
        if(verbose): print 'PASS RIM'
        ci_condition = 1;

    return flag_CI,ci_condition,nest, alge_trace

class AMOS():
    def __init__(self, k=2, Kmax=50, alpha=0.05,
                beta=0.05,pth=10^-5,Kmeans_replicate=50, n_jobs=1,
                verbose=False):
        """ Automated model order selection (AMOS) algorithm for spectral graph clustering (SGC)

        Parameters
        ----------
            k -  number of clusters - start with 2 partitions
            Kmax -  maximum number of allowed clusters, smaller value of Kmax can speed up the computation process.
            alpha - confidence interval parameter for homogeneous RIM test
            beta - confidence interval parameter for inhomogeneous RIM test
            pth - significance level for V-test
            Kmeans_replicate - Number of time the k-means algorithm will be run. 
            verbose - whether to print intermediate results
            n_jobs - the number of jobs to run in parallel. If -1, then the number of jobs is set to the number of cores.
        """

        self.k = k
        self.Kmax = Kmax
        self.alpha = alpha
        self.beta = beta
        self.pth = pth
        self.Kmeans_replicate = Kmeans_replicate
        self.verbose = verbose
        self.n_jobs = n_jobs

        
    def check_matrix(self):
        """Evaluation of the input matrix A
        """

        # check if the matrix is a Scipy sparse Matrix
        if not issparse(self.Aconn):
            self.Aconn = csr_matrix(self.Aconn)
        # check symmetry
        C = self.Aconn-self.Aconn.transpose()
        err = normSparse(C)
        if(err > 1e-10): 
            raise NameError('adjacency matrix is not symmetric')
        # check connectivity   
        num_comp = connected_components(self.Aconn)[0]
        if(num_comp != 1): 
            raise NameError('graph is not connected')
        # check nonnegativity
        if(self.Aconn[self.Aconn < 0.0].shape[1] > 0): 
            raise NameError('negative weight detected')

        
    def get_eig_vector_Kmax(self,sigma=10e-10):
        """ Compute Kmax smallest eigenpairs

        Parameters
        ----------
        sigma - shift regularizer 

        Returns
        ---------
        Acc - Regularized adjacency matrix
        eigVector_Kmax - Kmax eigenvalues
        """

        self.check_matrix()
        
        n = self.Aconn.shape[0]
        if(n < self.Kmax): 
            raise NameError('Kmax is too high (> n)')
    
        d = self.Aconn.sum(axis=0)
        Dinv = spdiags((1/np.sqrt(d)).reshape(-1),0,n,n)
        # regularized adjacency matrix Acc
        Acc = Dinv.dot(self.Aconn.dot(Dinv));
        Acc = np.add(Acc,np.transpose(Acc))/2
        
        # Graph Laplacian matrix of the regularized adjacency matrix
        D = spdiags(Acc.sum(axis=0),0,n,n,format='csr'); 
        L = np.add(D, -Acc);

        eigenvalues, eigVector_Kmax  = eigsh(np.add(eye(n).multiply(sigma),L),k=self.Kmax,which='SM')
        eigValue_Kmax = np.add(eigenvalues, -sigma*eye(self.Kmax).diagonal()).clip(min=0)

        # sorted eigenvalues and eigenvectors in ascending order
        idx = eigValue_Kmax.argsort()
        eigVector_Kmax = eigVector_Kmax[:,idx]
        return Acc, eigVector_Kmax

    def predict(self,Aconn):
        """ Predict the number of clusters

        Parameters
        ----------
            Aconn - Adjacency matrix of an undirected, weighted, and connected graph, which is a Scipy sparse matrix.
          
        Returns:
        ---------
            Number of clusters K,
            Identified clusters (labels)
        """
    
        num_pass_cases = 1 #stop at the minimal partition that satisfies the phase transition analysis
        flag=1; flag_CI=1; num_pass=0;

        self.Aconn = Aconn

        Acc, eigVector_Kmax = self.get_eig_vector_Kmax()
        
        k = self.k
        west= 0;tLB=0;tUB = 0;
        while(flag):
            # Obtain K clusters with spectral clustering
            if(self.verbose): print('k=%i clusters'%k)
            eigVector_k = eigVector_Kmax[:,1:k]

            kmeans = skKmeans(n_clusters=k, init='k-means++', n_init=self.Kmeans_replicate,n_jobs=self.n_jobs,copy_x=False)
            labels = kmeans.fit_predict(eigVector_k)

            # local homogeneity testing
            flag_CI,ci_condition,nest,alge_trace = rimTest(self.Aconn,labels,k,Acc,self.pth,n_jobs=self.n_jobs,verbose=self.verbose)
            # Confidence Interval of homogeneous RIM for k=2
            if(flag_CI):
                # confidence interval  

                CI_ind,mest,pest,Pest = confidence_interval_spec_new(self.Aconn,k,labels,nest,self.alpha,self.verbose)
                tLB, tUB, West, west = spectral_bound_est_weighted(Acc,labels,nest,k,alge_trace)
                # case k >= 3
                if k >= 3:
                # 'k >3'
                    ci_condition = CI_ind;
                    if(self.verbose): print 'ci_condition',ci_condition
                    if(ci_condition):
                        if(self.verbose): print 'within ',str(100*(1-self.alpha)),' % confidence interval, HomRIM'         
                        flag_InRIM=0;
                    else:
                        if(self.verbose): print 'NOT within ',str(100*(1-self.alpha)),' % confidence interval, HomRIM'
                        flag_InRIM=1;
                        
                # homogeneous RIM phase transition test
                if(ci_condition):
                    test = west*pest
                    if(test<tLB):    
                        #  the rows of each Yk is identical and cluster-wise distinct such that SGC can be successful
                        if(self.verbose): print 'relieble'
                        if(k==2):
                            num_pass=num_pass+1;
                            if(num_pass==num_pass_cases):
                                flag=0;
                            # k>=3 
                        elif(ci_condition):
                            num_pass=num_pass+1;
                            if(num_pass==num_pass_cases):
                                flag=0;
                    elif(test<tUB):
                        if(self.verbose): print 'intermediate'              
                    else:
                        # the row sum of each Yk is zero, and the incoher- ence of the entries in Yk make it impossible for SGC to separate the clusters
                        if(self.verbose): print 'unreliable'
                elif(flag_InRIM):
                        # inhomogeneous RIM test
                        CI_ind_InRIM = confidence_interval_inhom_RIM_Anscombe_InhomWeight(self.beta,West,Pest,tLB,k,labels,n_jobs=self.n_jobs,verbose=self.verbose);
                        if(self.verbose): print 'Anscombe'
                        if(CI_ind_InRIM):
                            flag=0;
                            if(self.verbose): print 'within ',str(100*(1-self.beta)),' % confidence interval, InHomRIM'        
                        else:
                            if(self.verbose): print 'NOT within ',str(100*(1-self.beta)),' % confidence interval, InHomRIM'
            if(k==self.Kmax):
                flag=0
                warnings.warn('k meets Kmax')
            k+=1;
        # Output : number of clusters K and identified clusters (labels)
        self.best_k = k-1
        self.labels = labels
        return k-1, labels

    def get_labels(self):
        """ 
        Returns:
            Identified clusters
        """
        if hasattr(self,'best_k'):
             return self.labels
        else:
             warnings.warn('Amos needs to first predict the best K')
        
    def get_k(self):
        """ 
        Returns:
            Number of clusters K found
        """ 
        if hasattr(self,'best_k'):
             return self.best_k
        else:
             warnings.warn('Amos needs to first predict the best K')