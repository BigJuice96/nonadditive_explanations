import os
os.chdir('/Users/abdallah/Desktop/Kings College Project/Code/GIT_Repo')
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sksurv.column import encode_categorical
from sksurv.svm import FastKernelSurvivalSVM
from sklearn.preprocessing import normalize
import math
from sksurv.metrics import concordance_index_censored
from itertools import chain, combinations
from random import sample
from sklearn.kernel_approximation import Nystroem
import seaborn as sns
from sklearn import set_config

def data_transformation(X, q_additivity=None, pow_set = None):
    n, d = X.shape
    if q_additivity is None: q_additivity = d

    if pow_set is None:
        temp = range(0,d)
        pow_set = list(chain.from_iterable(combinations(temp, r) for r in range(q_additivity+1)))
        pow_set.remove(())
    X_hat = np.empty((n, len(pow_set)))
    for i in range(len(pow_set)):
        X_hat[:,i] = np.prod(X[:, pow_set[i]], axis=1)

    return X_hat, pow_set

def q_additive_innerproduct(x, z, q_additivity):
        arr = x * z
        d = len(arr)
        # a matrix to store all the steps in the dynamic programming 
        # Initialising all the values to 0
        dp = [ [ 0 for x in range(d)] for y in range(q_additivity)]
        
        # To store the answer for
        # current value of k
        sum_current = 0
        
        
        # For k = 1, the answer will simply
        # be the sum of all the elements
        for i in range(d):
            dp[0][i] = arr[i]
            sum_current += arr[i]

        inner_prod_k = np.ones((q_additivity,)) * sum_current
        
        # Filling the dp table in bottom up manner
        for i in range(1 , q_additivity):
            # The sum of the row to be used for calculating the values in the next row
            temp_sum = 0
    
            for j in range( 0,  d):
    
                # We will subtract previously computed value
                # so as to get the sum of elements from j + 1
                # to n in the (i - 1)th row
                sum_current -= dp[i - 1][j]
    
                dp[i][j] = arr[j] * sum_current #((i-1) / i) * arr[j - 1] * sum_current
                temp_sum += dp[i][j]
            sum_current = temp_sum
            inner_prod_k[i:] += temp_sum

        inner_prod = inner_prod_k[-1]    
        return inner_prod# , inner_prod_k, dp

def Shapley_kernel(X1, X2, q_additivity=None, method='formula', feature_type='numerical'):
        sample_no1, feature_no = X1.shape
        sample_no2, _ = X2.shape
        kernel_mat = np.zeros((sample_no1,sample_no2))
        
        if q_additivity == None: q_additivity = feature_no
        assert q_additivity <= feature_no and q_additivity > 0, f"q-additivtiy parameter should be positive and less than the number of features, got {q_additivity}"

        if method == 'bf':
            print("brute-force metho")
            X1_hat, _ = data_transformation(X1, q_additivity)
            X2_hat, _ = data_transformation(X2, q_additivity)
            for i in range(0,sample_no1):
                for j in range(i, sample_no2):
                    kernel_mat[i,j] =  np.inner(X1_hat[i,:], X2_hat[j,:])
                    kernel_mat[j,i] =  kernel_mat[i,j]

        elif q_additivity == feature_no and method != 'dp':
            if feature_type == 'numerical':
                for i in range(0,sample_no1):
                    for j in range(0, sample_no2):
                        kernel_mat[i,j] =  -1 + np.prod( (X1[i,]*X2[j,]) + 1)
                        #kernel_mat[j,i] =  kernel_mat[i,j]
            
            elif feature_type == 'binary':
                inner_prod = np.inner(X1, X2)
                kernel_mat = 2 ** inner_prod - 1

        else:
            if feature_type == 'numerical':
                for i in range(0,sample_no1):
                    for j in range(0, sample_no2):
                        kernel_mat[i,j] = q_additive_innerproduct(np.array(X1[i,]),np.array(X2[j,]), q_additivity)
                        #kernel_mat[j,i] =  kernel_mat[i,j]

                    #self.kernel_mat[j,i] = self.kernel_mat[i,j]
            elif feature_type == 'binary':
                for i in range(0,sample_no1):
                    for j in range(i, sample_no2):
                        count = np.inner(X1[i,],X2[j,])
                        in_prod = 0
                        for k in range(min(count, q_additivity)):
                            in_prod +=  math.comb(count, k+1)
                        
                        kernel_mat[i,j] = in_prod
                        kernel_mat[j,i] = in_prod

        return kernel_mat

def Shapley_value(X_full, alpha_hat_eta, sv_ind, q_additivity = None, method='dp', feature_type = 'numerical'):
 
    X = X_full[sv_ind,:]
    n, d = X.shape
    if q_additivity is None: q_additivity = d

    val = np.zeros((d,))

    ######## Brute-force calculation of Shapley value
    if method == 'bf':
        X_hat, pow_set = data_transformation(X, q_additivity)
        weights = np.array([1/len(p) for p in pow_set], dtype=float)
        X_hat_weighted = X_hat * weights

        for i in range(d):
            #weight_i = weights[idx]
            #X_hat_weighted = X_hat[:,idx] * weight_i
            idx = [idx for idx, pset in enumerate(pow_set) if set(pow_set[i]).issubset(pset)]
            X_hat_weighted_i = X_hat_weighted[:,idx]
            omega_bf = np.sum(X_hat_weighted_i, axis=1)
            val[i] = np.inner(omega_bf, alpha_hat_eta)

    ######### Dynamic programming approach for computing Shapley value
    elif method == 'dp':
        for i in range(d):
            omega_dp, dp = Omega(X,i, q_additivity, feature_type=feature_type)
            val[i] = np.inner(omega_dp, alpha_hat_eta)
    else:
        raise Exception(f"The method should be either bf (brute-force) or dp (dynamic programming), given {method}")

    return val

def Omega(X, i, q_additivity= None, feature_type='numerical'):
    n, d = X.shape
    if q_additivity == None: q_additivity = d


    idx = np.arange(d)
    idx[i] = 0
    idx[0] = i
    X = X[:,idx]

    
    if feature_type == 'binary':
        print("binary feature type")
        omega = np.zeros((n,))
        ind_nonzeros = np.where(X[:,0] > 0)[0].tolist()
        for i in ind_nonzeros:
            xi_ones = np.where(X[i,1:] > 0)[0].tolist()
            xi_ones_count = len(xi_ones)
            temp = 0
            for j in range(1,q_additivity):
                temp += (1 / (j+1)) * (math.comb(xi_ones_count,j)) 
            
            omega[i] = temp 
        omega[ind_nonzeros] = (1 + omega[ind_nonzeros])
        return omega, None
      
    dp = np.zeros((q_additivity, d, n))
        
    # To store the answer for
    # current value of k
    sum_current = np.zeros((n,))
    
    
    # For k = 1, the answer will simply
    # be the sum of all the elements
    for i in range(d):
        dp[0,i,:] = X[:,i]
        sum_current += X[:,i]

    
    # Filling the dp table in bottom up manner
    for i in range(1 , q_additivity):
        # The sum of the row to be used for calculating the values in the next row
        temp_sum = np.zeros((n,))

        for j in range( 0,  d):

            # We will subtract previously computed value
            # so as to get the sum of elements from j + 1
            # to n in the (i - 1)th row
            sum_current -= dp[i - 1,j,:]

            dp[i,j,:] = (i / (i+1)) * X[:,j] * sum_current #((i-1) / i) * arr[j - 1] * sum_current
            temp_sum += dp[i,j,:]
        sum_current = temp_sum

    omega = np.sum(dp[:,0,:],axis=0)
    return omega, dp

    
def score_survival_model(model, X, y):
    prediction = model.predict(X)
    result = concordance_index_censored(np.array(pd.DataFrame(y).iloc[:,0]), np.array(pd.DataFrame(y).iloc[:,1]), prediction) # weird to index void arrays, so I convert it to a pd df and then select the column I want as a np.array
    return result[0]


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


    



def nonadditive_explainer(data_x, y, method = "SurvMLeX", alpha=1.0, finetune=True, rank_ratio=1.0, fit_intercept=False, n_shuffle_splits=5, random_state=None, range_alpha= (2.0 ** np.arange(-8, 7, 2))):
    """
    Takes a dataframe with survival targets and returns Shapley values indicating the feature importance for each feature.
    data_x is a pandas dataframe and y are the survival targets.
    Method can either be "SurvMLeX" or "SurvChoquEx"
    returns a pandas dataframe.
    If finetune is set to true, it returns the optimal alpha.
    """
    set_config(display="text")  # displays text representation of estimators
    sns.set_style("whitegrid")  
    
    x = encode_categorical(data_x)
    x_normalized = normalize(x.values, norm='l2')
    if method == "SurvMLeX":
        kernel_matrix = Shapley_kernel(x_normalized,x_normalized) #clinical_kernel(data_x) 
    elif method == "SurvChoquEx":
        kernel_matrix = ChoquetKernel(x_normalized) # This line is different 
        kernel_matrix = kernel_matrix.get_kernel()
    else: 
        assert("No method specified: can be either SurvMLeX or SurvChoquEx")
    param_grid = {"alpha": range_alpha}
    cv = ShuffleSplit(n_splits=n_shuffle_splits, test_size=0.5, random_state=random_state)
    kssvm = FastKernelSurvivalSVM(alpha=alpha, optimizer="rbtree", kernel="precomputed", random_state=0, rank_ratio=rank_ratio, fit_intercept=fit_intercept)

    if finetune==True:
        kgcv = GridSearchCV(kssvm, param_grid, scoring=score_survival_model, n_jobs=-1, refit=False, cv=cv)
        kgcv = kgcv.fit(kernel_matrix, y)
        alpha = kgcv.best_params_["alpha"]
    kssvm = FastKernelSurvivalSVM(alpha=alpha, optimizer="rbtree", kernel="precomputed", random_state=0, rank_ratio=rank_ratio, fit_intercept=fit_intercept)
    kssvm = kssvm.fit(kernel_matrix, y)    

    ## Computing shapley value
    if method == "SurvMLeX":
        shap_vals = Shapley_value(x_normalized, kssvm.coef_, np.arange(0,data_x.shape[0])) 
    elif method == "SurvChoquEx":
        shap_vals = Shapley_value_Choquet(x_normalized, kssvm.coef_, "") #137 is the number of data points

    result = pd.DataFrame()
    result["shapley_values"]=shap_vals
    result["shapley_values_absolute"]=np.absolute(shap_vals)
    result["feature_names"]= list(data_x.columns)
    result = result.sort_values(by="shapley_values_absolute", ascending=False)
    del result["shapley_values_absolute"]
    return result







