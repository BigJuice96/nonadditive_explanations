import os
os.chdir('/Users/abdallah/Desktop/Kings College Project/Code/GIT_Repo')
import numpy as np
import pandas as pd
import math
from sksurv.metrics import concordance_index_censored
from itertools import chain, combinations

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






# ###################### 

# set_config(display="text")  # displays text representation of estimators
# sns.set_style("whitegrid")

# data_x, y = load_veterans_lung_cancer()

# x = encode_categorical(data_x)

# x_normalized = normalize(x.values, norm='l2')
# kernel_matrix = Shapley_kernel(x_normalized,x_normalized) #clinical_kernel(data_x)
# param_grid = {"alpha": 2.0 ** np.arange(-12, 13, 2)}
# cv = ShuffleSplit(n_splits=100, test_size=0.5, random_state=0)

# kssvm = FastKernelSurvivalSVM(optimizer="rbtree", kernel="precomputed", random_state=0)
# kssvm = kssvm.fit(kernel_matrix, y)

# kgcv = GridSearchCV(kssvm, param_grid, scoring=score_survival_model, n_jobs=1, refit=False, cv=cv)
# kgcv = kgcv.fit(kernel_matrix, y)
# alpha = pd.DataFrame(kgcv.cv_results_).loc[pd.DataFrame(kgcv.cv_results_)["rank_test_score"] ==1,['param_alpha']].values[0][0]

# kssvm = FastKernelSurvivalSVM(alpha=alpha, optimizer="rbtree", kernel="precomputed", random_state=0)
# kssvm = kssvm.fit(kernel_matrix, y)


# ## Computing shapley value
# shap_vals = Shapley_value(x_normalized, kssvm.coef_, np.arange(0,137)) #137 is the number of data points
# print(shap_vals)


# ############################



# set_config(display="text")  # displays text representation of estimators
# sns.set_style("whitegrid")

# data_x, y = load_veterans_lung_cancer()

# x = encode_categorical(data_x)

# x_normalized = normalize(x.values, norm='l2')
# kernel_matrix = choquet.ChoquetKernel(x_normalized) #clinical_kernel(data_x) # TODO He will give me another function to get Choquet kernel (only give it one x_normalised: choquet_kernel(x_normalised))
# kernel_matrix = kernel_matrix.get_kernel()

# param_grid = {"alpha": 2.0 ** np.arange(-12, 13, 2)}
# cv = ShuffleSplit(n_splits=100, test_size=0.5, random_state=0)

# kssvm = FastKernelSurvivalSVM(optimizer="rbtree", kernel="precomputed", random_state=0)
# kssvm = kssvm.fit(kernel_matrix, y)

# kgcv = GridSearchCV(kssvm, param_grid, scoring=score_survival_model, n_jobs=1, refit=False, cv=cv)
# kgcv = kgcv.fit(kernel_matrix, y)
# alpha = pd.DataFrame(kgcv.cv_results_).loc[pd.DataFrame(kgcv.cv_results_)["rank_test_score"] ==1,['param_alpha']].values[0][0]

# kssvm = FastKernelSurvivalSVM(alpha=alpha, optimizer="rbtree", kernel="precomputed", random_state=0)
# kssvm = kssvm.fit(kernel_matrix, y)


# ## Computing shapley value
# shap_vals_choqu = choquet.Shapley_value_Choquet(x_normalized, kssvm.coef_, "") #137 is the number of data points
# print(shap_vals_choqu)



# ############################