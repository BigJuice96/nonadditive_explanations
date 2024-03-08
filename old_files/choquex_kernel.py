from random import sample
import numpy as np
import math
import os
from sklearn.kernel_approximation import Nystroem
from itertools import chain, combinations

class ChoquetKernel:
    def __init__(self, X, k_additivity=None):
        self.X = X
        self.sample_no, self.feature_no = X.shape
        if k_additivity == None: k_additivity = self.feature_no
        self.k = np.min((k_additivity, self.feature_no))

    def get_kernel(self, type='brute-force'):
        choq_product = []
        choq_ker = np.zeros((self.sample_no, self.sample_no))
        self.X_sorted = None
        self.X_sort_index = None
        if type == 'min':
            choq_product = ChoquetInnerProduct(feature_no=self.feature_no, type='min')
            X_sorted = np.sort(X, axis=1)
            X_sort_index = np.argsort(X, axis=1)
            X_sortindex_sorted = np.argsort(X_sort_index, axis=1)
            for i in range(self.sample_no):
                for j in range(i, self.sample_no):
                    choq_ker[i,j] = choq_product.choq_product(x=X[i,:], y=X[j,:], xs = X_sorted[i,:], xi = X_sort_index[i,:])

            self.X_sorted = X_sorted
            self.X_sort_index = X_sort_index

        elif type == 'brute-force':
            choq_product = ChoquetInnerProduct(feature_no= self.feature_no, type='brute-force')
            data, _ = mobius_transformation(self.X, k=self.k)

            for i in range(self.sample_no):
                for j in range(i, self.sample_no):
                    choq_ker[i,j] = choq_product.choq_product(data[i,:], data[j,:])

        elif type == 'binary':
            inner_prod = np.inner(self.X, self.X)
            choq_ker = 2 ** inner_prod - 1
            #choq_product = ChoquetInnerProduct(feature_no= self.feature_no, type='binary', k_additivity=self.k)
            #data = mobius_transformation(self.X, k=self.k)

            #kernel_approx = Nystroem(choq_product.choq_product, n_components= 1000)
            #gram = kernel_approx.fit_transform(self.X)

            #choq_ker = np.inner(gram, gram)

            #for i in range(self.sample_no):
            #    for j in range(i, self.sample_no):
            #        choq_ker[i,j] = choq_product.choq_product(self.X[i,:], self.X[j,:])
        
        return choq_ker + choq_ker.T


class ChoquetInnerProduct:
    def __init__(self, feature_no, type, k_additivity=None):
        if k_additivity is None or k_additivity < 0:
            k_additivity = feature_no
        self.feature_no = feature_no
        self.type = type
        self.k_additivity = np.min((k_additivity, feature_no))

    def choq_product(self, x, y, xs=None, xi=None):
        if self.type == 'min':
            return self.min_inner(x,y, xs, xi)
        elif self.type == "brute-force":
            return self.inner_product(x, y)
        elif self.type == "binary":
            return self.binary_product(x, y, self.k_additivity)

    def inner_product(self, x, y):
        return np.inner(x, y)

    def min_inner(self, x, y, xs, xi):
        yp = y[xi].squeeze()
        # yi = np.argsort(y)
        # yss = np.argsort(yi)
        
        # yp_s = yss[xi]

        idx = np.ones((self.feature_no,)) == 1
        in_p = np.inner(x, y)


        for i in range(self.feature_no-1):
            y_iter_sorted = np.sort(yp[idx]) # ys[idx][yi[idx]] #  

            pivot_index = np.where(y_iter_sorted == yp[i])[0][0]
            y_iter_sorted[pivot_index:] = yp[i]
            y_iter_sorted = y_iter_sorted[0:-1]
            #if y_iter_sorted[0] == ys[i]:
            #    y_iter_sorted = ys[i]* np.ones(y_iter_sorted.shape)

            inn_val = np.inner(2 ** np.array((range(self.feature_no-i-2,-1,-1))), y_iter_sorted) 
            in_p += xs[i]*inn_val

            idx[i] = False
            #yi -= 1
            #yi -= np.min(yi)

        return in_p

    def binary_product(self, x, y, k_additivity):
        count = np.count_nonzero(x * y)
        in_prod = 0
        for k in range(min(count, k_additivity)):
            in_prod +=  math.comb(count, k+1)

        return in_prod

def mobius_transformation(X, k=None, pow_set = None):
    sample_no, feature_no = X.shape
    if k is None:
        k = feature_no
    if pow_set is None:
        temp = range(0,feature_no)
        pow_set = list(chain.from_iterable(combinations(temp, r) for r in range(k+1)))
        pow_set.remove(())
    
    data_mobius = np.empty((sample_no, len(pow_set)))

    for i in range(len(pow_set)):
        data_mobius[:,i] = np.min(X[:, pow_set[i]], axis=1)

    return data_mobius, pow_set

def Shapley_value_Choquet(X, alpha, type):
    #if type == 'brute-force':
    n, d = X.shape
    val = np.zeros((d,))

    X_hat, pow_set = mobius_transformation(X)
    weights = np.array([1/len(p) for p in pow_set], dtype=float)
    X_hat_weighted = X_hat * weights

    for i in range(d):
        #weight_i = weights[idx]
        #X_hat_weighted = X_hat[:,idx] * weight_i
        idx = [idx for idx, pset in enumerate(pow_set) if set(pow_set[i]).issubset(pset)]
        X_hat_weighted_i = X_hat_weighted[:,idx]
        omega_bf = np.sum(X_hat_weighted_i, axis=1)
        val[i] = np.inner(omega_bf, alpha)

    return val


if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.svm import SVC

    np.random.seed(42)

    breast_cancer =  datasets.load_breast_cancer()
    X = breast_cancer['data']
    y = breast_cancer['target']
    y[y == 0] = -1

    n=100
    X = np.random.rand(n, 4) * 7 # X.data
    y = np.random.choice([-1, 1], size=n)

    ## Use the following two lines to create the kernel matrix
    choquet_kernel = ChoquetKernel(X)
    K1 = choquet_kernel.get_kernel(type='min')
    
    #K2 = choquet_kernel.get_kernel(type='brute-force')

    #print("The norm 2 difference of two kernel is: " + str(np.linalg.norm(K1-K2)))
    #print("The maximum difference between the two kernels is: " + str(np.max(K1-K2)))
    

    svc = SVC(kernel='precomputed')
    svc.fit(K1,y)

    alpha = svc.dual_coef_ # for survival analysis: svsvr.coef_ ; the size is #samples
    svs = svc.support_ # support vectors the index of alpha that are non-zero

    X_support = X[svs, :]
    y_support = y[svs]

    alpha_hat = (y_support * alpha).flatten()

    val = Shapley_value_Choquet(X_support, alpha_hat, type='brute-force') # Shapley_computation(X, kssvm.coef_)
    
    print("DONE!")


    