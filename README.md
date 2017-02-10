# [AMOS](http://arxiv.org/abs/1604.03159): AN AUTOMATED MODEL ORDER SELECTION ALGORITHM FOR SPECTRAL GRAPH CLUSTERING

##### By [Pin-Yu Chen](mailto:pinyu@umich.edu), [Thibaut Gensollen](mailto:thibaut.gensollen@ens-cachan.fr), and [Alfred O. Hero III](mailto:hero@umich.edu), Fellow, IEEE
###### Department of Electrical Engineering and Computer Science, University of Michigan, Ann Arbor, USA

One of the longstanding problems in spectral graph clustering (SGC) is the so-called model order selection problem: automated selection of the correct number of clusters. This is equivalent to the problem of finding the number of connected components or communities in an undirected graph. In this paper, we propose AMOS, an automated model order selection algorithm for SGC. Based on a recent develop- ment of clustering reliability for SGC under the random interconnec- tion model, AMOS works by incrementally increasing the number of clusters, estimating the quality of identified clusters, and providing a series of clustering reliability tests. Consequently, AMOS outputs clusters of minimal model order with statistical clustering reliabil- ity guarantees. Comparing to three other automated graph clustering methods on real-world datasets, AMOS shows superior performance in terms of multiple external and internal clustering metrics.

### Requirements :

AMOS is using some external libraries for working, be sure to install them before :

- [Scipy](https://www.scipy.org/)
- [Sklearn](http://scikit-learn.org/)
- [Numpy](http://www.numpy.org/)
- [Joblib](https://pythonhosted.org/joblib/)
- [Sympy](http://www.sympy.org/fr/index.html)

### Running AMOS

```sh
$ python examples.py --A=hibernia
```

```python
import amos
model = amos.AMOS(k_init,Kmax)
best_k, labels = model.predict(A)
```

You can set some parameters:
- A (required), the adjacency matrix of an undirected, weighted, and connected graph. The matrix is then converted to a Scipy Sparse Matrix (csr).
- k_init (optional, Default : 2), number of clusters - start with 2 partitions by default.
- Kmax (optional, Default : 10), maximum number of allowed clusters, smaller value of Kmax can speed up the computation process.
- Kmeans_replicate (optional), Number of time the k-means algorithm will be run. (Default : 50).

- alpha (optional), confidence interval parameter for homogeneous RIM test (Default : 0.05)
- beta (optional), confidence interval parameter for inhomogeneous RIM test (Default : 0.05)
- pth (optional), significance level for V-test (Default : 10^-5)

- n_jobs (optional), the number of jobs to run in parallel. If -1, then the number of jobs is set to the number of cores. (Default : 1).
- verbose (optional), for displaying some processing informations.

##### Outputs :

- Number of clusters K,
- Identified clusters (labels)


#### Multiple examples datasets :

- hibernia, should find 2
- cogent, should find 3
- ieee, should find 4
- [facebook](https://snap.stanford.edu/data/egonets-Facebook.html), should find 5
- minnesota, should find something between 45 and 55

More details are accessible [here](http://arxiv.org/abs/1604.03159).

#### Running Tests

A test file `test.py` will test the examples to be sure the code is working properly.


#### Usage

Please cite the arXiv and the ICASSP papers when using the code.

License
----

MIT