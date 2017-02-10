import unittest

import numpy as np
import scipy.io
import scipy.sparse
import amos
import sys

alpha=0.05  # confidence interval parameter for homogeneous RIM test
beta=0.05  # confidence interval parameter for inhomogeneous RIM test
pth=10^-5 # significance level for Vest
 # maximum number of allowed clusters, smaller value of Kmax can speed up the computation process; otherwise Kmax can set to be n

Kmax = 4;
k_init = 2;
model = amos.AMOS(k_init,Kmax)

class TestAMOSParameters(unittest.TestCase):
    def test_Kmax_meets(self):
        A = scipy.io.loadmat('examples/Minnesota_road_adjacency_connected.mat')['A']
        
        best_k, labels = model.predict(A)
        self.assertEqual(best_k, 4)

    def test_cogent(self):
        A = scipy.sparse.csr_matrix(scipy.io.loadmat('examples/cogent_adjacency.mat')['A'])
        best_k, labels = model.predict(A)
        self.assertEqual(best_k, 4)

    def test_hibernia(self):
        A = scipy.sparse.csr_matrix(scipy.io.loadmat('examples/hibernia_adjacency.mat')['A'])
        best_k, labels = model.predict(A)
        self.assertEqual(best_k, 2)

    def test_ieee(self):
        A = scipy.sparse.csr_matrix(scipy.io.loadmat('examples/IEEERTS96Adjacency.mat')['A'])
        best_k, labels = model.predict(A)
        self.assertEqual(best_k, 3)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAMOSParameters)
    unittest.TextTestRunner(verbosity=3).run(suite)
