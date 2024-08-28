import pandas as pd
import numpy as np
from numpy.lib import recfunctions as rfn
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
from functools import partial
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sksurv.linear_model import CoxnetSurvivalAnalysis
import os
from lime import lime_tabular
import itertools
# from statsmodels.stats.outliers_influence import variance_inflation_factor
# import smote_variants as sv 
# from imblearn.over_sampling import SMOTENC




# Preprocessing

def standardise_features(df, feature_names):
    """
    Pass it a df and a list of feature labels. 
    Will return a df after standardising indicated features.
    """
    for feature_name in feature_names:
        df[feature_name] = (df[feature_name] - np.mean(np.array(df[feature_name])))/np.std(np.array(df[feature_name]))
    return df

def merge_dataframes(dataframes, on, remove_duplicate_columns = True):
    """
    Merges dataframes into one based on specified column.
    Removes duplicate columns.
    """
    # Merge dataframes
    merged_df = dataframes[0].copy()
    for dataframe in dataframes[1:]:
        merged_df = merged_df.merge(dataframe, on = on)

    if remove_duplicate_columns == True:
        # Remove duplicate columns
        for column in merged_df.columns:
            if "_y" in column:
                merged_df = merged_df.drop(columns=[column])
        # Rename columns without the _x
        for column in merged_df.columns:
            if "_x" in column:
                merged_df = merged_df.rename(columns={column: column[:-2]})
        # still some duplicates in here to remove: 
        merged_df = merged_df.loc[:,~merged_df.columns.duplicated()].copy()
    return merged_df



# def balance_data_smotenc(df, y, cat_features):
#     """
#     df will not include target (train_x).
#     y will be the target (0,1) , so train["dem_hse_w8"]
#     Rounds features that are supposed to be discrete.
#     Applies SMOTENC to df.
#     cat_features will be a list of the names of the categorical features
#     """
#     cat_boolean = []
#     for feature in list(df.columns):
#         if feature in cat_features:
#             cat_boolean.append(True)
#         else:
#             cat_boolean.append(False)  
#     smote_nc = SMOTENC(categorical_features=cat_boolean, random_state=0)
#     new_df, new_y = smote_nc.fit_resample(df, y)

#     for feature in df.columns:
#         if not any(df[feature] - np.round(df[feature],0) > 0):  # Rounds values up to create discrete values if original feature was discrete
#             new_df[feature] = np.round(new_df[feature],0)
#     return new_df, new_y


# def balance_dataset(dataframe , class_feature, categorical_features, balancer = sv.Safe_Level_SMOTE()):
#     """
#     Pass it the training data with all variables included (also the targets).
#     Will conduct oversampling to generate synthetic data and balance out the df.
#     Will then round categorical and discrete features so that the new instances are in line.
#     For SMOTENC:
#         df will not include target (train_x).
#         y will be the target (0,1) , so train["dem_hse_w8"]
#         Rounds features that are supposed to be discrete.
#         Applies SMOTENC to df.
#         cat_features will be a list of the names of the categorical features
#     """
#     if balancer == "SMOTENC":
#         df = dataframe.drop(class_feature, axis=1)
#         cat_boolean = []
#         for feature in list(df.columns):
#             if feature in categorical_features:
#                 cat_boolean.append(True)
#             else:
#                 cat_boolean.append(False)  
#         smote_nc = SMOTENC(categorical_features=cat_boolean, random_state=0)
#         new_df, new_y = smote_nc.fit_resample(df, dataframe[class_feature])

#         for feature in df.columns:
#             if not any(df[feature] - np.round(df[feature],0) > 0):  # Rounds values up to create discrete values if original feature was discrete
#                 new_df[feature] = np.round(new_df[feature],0)
        
#         new_df[class_feature] = np.array(new_y)
#         return new_df

#     X = np.array(dataframe)
#     y = np.array(dataframe[class_feature])
#     oversampler = balancer

#     # X_samp and y_samp contain the oversampled dataset
#     X_samp, y_samp= oversampler.sample(X, y)
#     dataframe = pd.DataFrame(X_samp, columns=list(dataframe.columns))
#     dataframe[class_feature] = y_samp

#     # Have to round features that are categorical/discrete
#     non_cat_features = [x for x in list(dataframe.columns) if x not in  categorical_features]
#     discrete_feature_indices = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,21] # I checked for those manually...
#     for feature in dataframe.columns:
#         if feature in categorical_features:
#             dataframe[feature] = np.round(dataframe[feature],0)
#         elif feature in [non_cat_features[x] for x in discrete_feature_indices]:
#             dataframe[feature] = np.round(dataframe[feature],0)
#     return dataframe



def detect_nan_features(df, threshold = 0.3, print_features = True):
    """
    Prints features in dataframe that consist of a lot of NaNs.
    """
    nan_features = []
    for feature in df.columns:
        if sum(np.isnan(df[feature])) / df.shape[0] > threshold:
            
            if print_features == True:
                print(feature, " consists of ", (sum(np.isnan(df[feature])) / df.shape[0])*100, "% NaNs.")
            nan_features.append(feature)
    return nan_features


def convert_time_y_to_surv_y(df = None, y = None, event_indicator = None, only_targets = False, round_y= False): 
    """
    Splits survival dataframe into x and y dataframes.
    Pass y and event_indicator as strings. 
    Function turns time-to-event column to a survival outcome column suited for sci-kit survival models (boolean+float).
    Assumes no data is censored.
    Removes outcome columns from training set
    """
    if round_y == True:
        y = np.round(y,0)
    if only_targets == False:
        surv_y1 = np.array(df[event_indicator]).astype("bool")
        surv_y2 = np.array(df[y])
        surv_y = rfn.merge_arrays((surv_y1, surv_y2))
        x = df.drop([y,event_indicator], axis = 1)
        return x, surv_y
    else:
        event_indicator = np.full(len(y), True)
        surv_y1 = event_indicator
        surv_y2 = y
        surv_y = rfn.merge_arrays((surv_y1, surv_y2))
        return surv_y
        


def find_undefined_variables(all_dfs, all_columns, defined_variables):
    """
    Finds all variables that are in my dataframe but are not defined in the list of variable definitions.
    """
    variables_not_defined = []
    for feature in  all_columns:
        if not feature in  defined_variables:
            print(feature)
            variables_not_defined.append(feature)
    for variable in variables_not_defined:
        for i in range(len(all_dfs)):
            if variable in list(all_dfs[i].columns):
                print(variable, " is from dataframe ", i)
    return variables_not_defined


# Prediction and evaluation


def aggregate_surv_function(surv_curves, timestamps, method = "percentile", threshold = 0.5):
    """
    Takes survival curves as arrays and returns event + time-to-event predictions as pandas df.
    Aggregates the survival functions of those cases that are predicted to have dementia according to integral or median method.
    All instances whose survival curve falls below threshold are predicted to have dementia. 
    Check this for the reason why we chose 0.5 as the cutoff:
    https://www.medcalc.org/manual/kaplan-meier.php#:~:text=The%20median%20survival%20is%20the,median%20time%20cannot%20be%20computed.
    """
    surv_curves = np.array(surv_curves)
    result = pd.DataFrame()
        # if the predicted curve does not fall below the threshold at any timepoint we will consider it censored
    if method == "median": threshold = 0.5
    boolean_indexer = [any(x < threshold) for x in surv_curves] 
    result["event"] = boolean_indexer 
        # Censored events will maintain a timepoint that corresponds to the last unique timestamp of the predicted survival functions
    result["time"] = np.full(result.shape[0], timestamps[-1], dtype="float") 
    if method == "trapezoid":
        result.loc["time"] = np.round(trapezoid(surv_curves),0)
    else:
        if method != "median":
            if method != "percentile": 
                assert("No method has been specified: all cases predicted maximum time-to-event.")
        result.loc[boolean_indexer, "time"] = [np.argmax(x < threshold) for x in surv_curves[boolean_indexer]] # index for which probabiity falls below threshold
    result.loc[boolean_indexer,"time"] = [timestamps[x] for x in np.array(result.loc[boolean_indexer,"time"], dtype="int")] # use index from line before to get the corresponding timestamp
    return result 



# def plot_stepfunction(stepfunction):
#     for i, s in enumerate(stepfunction):
#         plt.step(rsf.event_times_, s, where="post", label=str(i))
#     plt.ylabel("Survival probability")
#     plt.xlabel("Time in days")
#     plt.grid(True)
#     plt.show()
#     pass


def assess_residuals(residuals, show_plot = True, print_stats=True):
    """
    Assesses the residuals of the time-to-event predictions by checking whether they center around 0 or have a systematic bias.
    """
    if print_stats==True:
        print("Median: ",np.median(residuals),"\nMean: ",np.mean(residuals),"\nStandad Deviation: ",np.std(residuals),"\n")
    if show_plot == True:
        plt.hist(residuals)
        plt.show()
    
    residuals_stats=dict()
    residuals_stats["median"]=np.median(residuals)
    residuals_stats["mean"]=np.mean(residuals)
    residuals_stats["std"]=np.std(residuals)
    return residuals_stats


def calculate_roc_auc(targets, y_pred_proba):
    """
    Given probabilities and targets gives the AUC metric
    """
    auc = metrics.roc_auc_score(targets, y_pred_proba)
    # Recall that a model with an AUC score of 0.5 is no better than a model that performs random guessing.
    return auc




def sensitivity_specificity(target_bool, prediction_bool, print_results = True):
    """
    Must pass prediction and target as numpy arrays. 
    """
    false_pos = 0
    false_neg = 0
    true_pos = 0
    true_neg = 0
    for i in range(len(prediction_bool)-1):
        if prediction_bool[i] == True and target_bool[i] == True:
            true_pos += 1
        elif prediction_bool[i] == False and target_bool[i] == False:
            true_neg += 1
        elif prediction_bool[i] == True and target_bool[i] == False: 
            false_pos += 1
        elif prediction_bool[i] == False and target_bool[i] == True:
            false_neg += 1
        else:
            print("Case ", i, " not accounted for.")
    result = dict()
    result["Accuracy"] = (true_pos + true_neg)/(len(prediction_bool))
    result["hits"] = true_pos + true_neg
    result["misses"] = false_neg + false_pos
    result["false_pos"] = false_pos
    result["true_pos"] = true_pos
    result["false_neg"] = false_neg
    result["true_neg"] = true_neg
    if (true_pos+false_pos) > 0:
        result["precision"] =  true_pos/(true_pos+false_pos)
    else:
        result["precision"] =  np.nan
    if (true_pos+false_neg) > 0:
        result["sensitivity"] = true_pos/(true_pos+false_neg)
    else:
        result["sensitivity"] = np.nan
    if np.isnan(result["precision"]):
        result["fscore"] = np.nan
    elif np.isnan(result["sensitivity"]):
        result["fscore"] = np.nan
    elif (result["precision"] + result["sensitivity"])>0:
        result["fscore"] = 2 * (result["precision"] * result["sensitivity"]) / (result["precision"] + result["sensitivity"])
    else: 
        result["fscore"] = np.nan

    if (true_neg+false_pos)>0:
        result["specificity"] = true_neg/(true_neg+false_pos)
    else:
        result["specificity"] = np.nan
    if print_results == True:
        print("Accuracy: ", (true_pos + true_neg)/(len(prediction_bool)))
        print("Hits: ", true_pos + true_neg)
        print("Misses: ", false_neg + false_pos)
        print("False positives: ", false_pos)
        print("False negatives: ", false_neg)
        print("Precision/PPV: ", result["precision"])
        print("Sensitivity/Recall: ", true_pos/(true_pos+false_neg))
        print("F-score: ",result["fscore"] )
        print("Specificity: ", true_neg/(true_neg+false_pos))
    return result



def plot_calibration(survival_curves, targets, title="calibration", n_bins=50, save_plot=False, show_plot=True):
    """
    plots calibration and returns predicted and observed y (binned).
    input:
        survival curves are the predictions on the data.
        df includes the target
    """
    calibration_df = pd.DataFrame()
    calibration_df["observed"] = targets
    if len(survival_curves.shape) == 1:
        y_pred = survival_curves
        calibration_df["predicted"] = y_pred
    else:
        y_pred = 1-survival_curves
        y_pred = y_pred[:,-1]
        calibration_df["predicted"] = y_pred
    calibration_df = calibration_df.sort_values(by=["predicted"])
    bins = np.array_split(calibration_df, n_bins)
    y_predicted_bins = []
    y_observed = []
    for i in range(len(bins)):
        y_predicted_bins.append(np.mean(bins[i]["predicted"]))
        y_observed.append(np.mean(bins[i]["observed"]))
    plt.plot(y_predicted_bins, y_observed)
    plt.plot([0,0.5,1], [0,0.5,1])
    plt.title(title)
    plt.xlabel("y_predicted")
    plt.ylabel("y_observed")
    if save_plot==True:
        results_dir = '/Users/abdallah/Desktop/Kings College Project/Code/Data prepation ELSA Dementia/DATA/calibration_plots/'
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        plt.savefig(results_dir + title)
    if show_plot==True:
        plt.show()
    return y_predicted_bins, y_pred

def calibration_rf_clf(train):
    """
    plots calibration of a random forest classifier on the data, using part of it for validation
    """
    t, v = train_test_split(train, test_size=0.2,  random_state=20)
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(t.drop(["dem_hse_w8","time"],axis=1), t["dem_hse_w8"])
    score = clf.score(v.drop(["dem_hse_w8","time"],axis=1), v["dem_hse_w8"])
    plot_calibration(clf.predict_proba(v.drop(["dem_hse_w8","time"],axis=1))[:,1], v["dem_hse_w8"], n_bins=10)
    return score


def calculate_fidelity(event_times1, event_times2, curve1, curve2, new_curve_shape = (18,)):
    if len(event_times1) == len(event_times2):
        if all(event_times1 == event_times2):
            fidelity = np.mean(np.sqrt(np.array((curve1-curve2)**2).astype(float)))
            return fidelity, None, None
    print("Event times differ: arrays must be reshaped.")
    event_times_curves = dict()
    event_times_curves[0] = event_times1
    event_times_curves[1] = event_times2
    curves = dict()
    curves["old1"] = curve1 #COX
    curves["old2"] = curve2 #RSF
    for i,current_curves in enumerate(list(curves.keys())):
        event_times = event_times_curves[i]
        new_curve = np.ones(new_curve_shape)
        old_curve = curves[current_curves]

        for time_point in np.arange(0,18, dtype="float"):
            if time_point not in event_times:
                if any(event_times > time_point):
                    new_curve[int(time_point)] = old_curve[event_times > time_point][0]
                elif any(event_times < time_point):
                    new_curve[int(time_point)] = old_curve[event_times < time_point][0]
            else:
                new_curve[int(time_point)] = old_curve[np.where(event_times == time_point)[0]]    
        curves[current_curves] = new_curve
    fidelity = np.mean(np.sqrt(np.array((curves["old1"]-curves["old2"])**2).astype(float)))
    return fidelity, curves["old1"], curves["old2"]


# Explanation


def find_categ_features(df): 
    """
    Ideally, pass a dummied dataframe (with dummied columns dtype set to "bool"; but should work in any case).
    Returns names of categorical columns in pandas dataframe.
    Will also return their indices.
    """
    cat_features = []
    if any(df.dtypes == "object"):
        cat_features_obj = df.dtypes[df.dtypes == "object" ].index
        cat_features.extend(cat_features_obj)
    if any(df.dtypes == "string"):
        cat_features_str = df.dtypes[df.dtypes == "string" ].index
        cat_features.extend(cat_features_str)
    if any(df.dtypes == "category"):
        cat_features_cat = df.dtypes[df.dtypes == "category" ].index
        cat_features.extend(cat_features_cat)
    if any(df.dtypes == "bool"):
        cat_features_bool = df.dtypes[df.dtypes == "bool" ].index
        cat_features.extend(cat_features_bool)
    if cat_features != []:
        cat_feat_indices = [list(df.columns).index(x) for x in cat_features]
        return cat_features, cat_feat_indices
    else:
        print("No categorical features found.")




# Interpretable Model

def order_hazard_ratios_by_importance(hazard_ratios):
    """Pass a pandas series as an argument with the feature names as indices"""
    array = np.absolute(np.array(1-hazard_ratios))
    order = list(array.argsort())
    order.reverse()
    return order


def kernel(d, kernel_width):
    """
    Used in calculate_weigths().
    """
    return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))


def calculate_weights(df_train, df_pert, kernel_multiplyer = 0.75):
    """
    Calculates weights using the LIME kernel.
    """
    kernel_width = float(np.sqrt(df_train.shape[1]) * kernel_multiplyer)
    kernel_fn = partial(kernel, kernel_width=kernel_width)
    distances = sklearn.metrics.pairwise_distances(
                np.array(df_pert),
                np.array(df_pert)[0].reshape(1, -1),
                metric='euclidean'
        ).ravel()    
    weights = kernel_fn(distances)
    return weights


# def find_high_VIF(df):    
#     """
#     Should adapt this to do it recursively and drop features until VIF is right.
#     Returns variables with a high VIF.
#     """
#     vif_data = pd.DataFrame()
#     vif_data["feature"] = df.columns
#     vif_data["VIF"] = [variance_inflation_factor(df.values, i)
#                             for i in range(len(df.columns))]
#     high_VIF_features = list(vif_data.loc[vif_data["VIF"] > 5,:]["feature"])
#     print(vif_data)
#     return high_VIF_features, vif_data


# def generate_pert_data(df, n_samples, categ_features, instance_index):
#     """
#     Takes pandas dataframe and generates n perturbed samples based on each features mean and std.
#     """
#     pert_data = pd.DataFrame()
#     for feature in df.columns:
#         pert_data[feature] = np.random.normal(np.mean(df[feature]), np.std(df[feature]), n_samples)
#         if not any(df[feature] < 0): # Makes feature only positive if original feature was > 0
#             pert_data[feature] = np.sqrt(pert_data[feature]**2)
#         if not any(df[feature] - np.round(df[feature],0) > 0):  # Rounds values up to create discrete values if original feature was discrete
#             pert_data[feature] = np.round(pert_data[feature],0)
#         if feature in categ_features:
#             pert_data[feature] = np.random.randint(0,2,n_samples).astype(float)
#     pert_data.iloc[0,:] = df.loc[instance_index,:]
#     return pert_data


def create_weighted_perturbed_data(data, 
                                   instance, 
                                   categorical_features, 
                                   binary=False, 
                                   lime_kernel=0.75,
                                   weigh_data=True,
                                   n=1000,
                                   kernel = "LIME",
                                   random_state = None
                                   ):

        # get indices of the categorical features for the following method
    x_categorical_indices = [ind for ind, x in enumerate(data.columns) if x in categorical_features] 
        # Use LIME to create perturbed samples
            # LIME's BaseDiscretizer class will discretize all features that are not listed as categorical
    expl = lime_tabular.LimeTabularExplainer(np.array(data), feature_names= list(data.columns), 
                                        categorical_features = x_categorical_indices, 
                                        random_state = random_state,
                                        verbose=True, mode='regression', discretize_continuous=True)
    data_initial, inverse = expl._LimeTabularExplainer__data_inverse(data_row = instance,
                        num_samples = n)  # first row of inverse corresponds to the instance to be explained
                    # Inverse is a matrix with real valued data consisting of perturbed samples
                    # Categorical features are perturbed by sampling from a distribution that reflects the frequencies of each ..
                    #... feature value (using sklearn.utils.check_random_state.choice )
                    # Continuous features are sampled from a normal distr. with mean and std of the original feature
                    # If discretize=True, continuous features are discretized so that data can be a binary df showing whether ...
                    #.... the feature value in each respective perturbed sample is the same or different
    scaled_data = (data_initial - expl.scaler.mean_) /expl.scaler.scale_ # This doesnt do anything if discretizer is True
        # Get weights for our data and weigh them by sampling with replacement

    distances = sklearn.metrics.pairwise_distances(
            scaled_data,
            scaled_data[0].reshape(1, -1),
            metric="euclidean"
    ).ravel()
    kernel_width = np.sqrt(data.shape[1]) * lime_kernel
    kernel_width = float(kernel_width)
    def kernel(d, kernel_width):
            return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))
    kernel_fn = partial(kernel, kernel_width=kernel_width)
    weights = kernel_fn(distances)

        # Get predictions for perturbed data from the blackbox model
    inverse = pd.DataFrame(inverse)
    inverse.columns = data.columns
        # Weigh the data
    if binary==True: #If true, we use the binary dataset for training. If not, we use the perturbations.
        scaled_data = pd.DataFrame(scaled_data)
        scaled_data.columns = list(inverse.columns)
    else:
        scaled_data = inverse.copy()
    if weigh_data==True:
        scaled_data = weigh_df(scaled_data, scaled_data.shape[0], weights)
    return scaled_data, inverse, weights

def round_discr_cat_features(original_df, new_df, categ_features):
    """
    All as pandas dataframes
    categ_features is a list of strings
    """
    for feature in original_df.columns:
        if not any(original_df[feature] < 0): # Makes feature only positive if original feature was > 0
            new_df[feature] = np.sqrt(new_df[feature]**2)
        if not any(original_df[feature] - np.round(original_df[feature],0) > 0):  # Rounds values up to create discrete values if original feature was discrete
            new_df[feature] = np.round(new_df[feature],0)
        if feature in categ_features: # Makes sure categorical features remain binary
            new_df[feature] = np.round(new_df[feature],0)
            new_df.loc[new_df[feature] == 1, feature] = new_df.loc[new_df[feature] == 1, feature] -1
    return new_df


def weigh_df(df, n_samples, weights):
    """
    Returns a weighted dataframe by sampling from it with replacement.
    Keeps the first row the same in both the input and output df.
    """
    instance = df.iloc[0,:]# first instance should remain instance of interest
    df = df.sample(n_samples, replace=True, axis = 0, weights = weights)
    df.iloc[0,:] = instance
    return df


def plot_coefficients(coefs, n_highlight):
    _, ax = plt.subplots(figsize=(9, 6))
    n_features = coefs.shape[0]
    alphas = coefs.columns
    for row in coefs.itertuples():
        ax.semilogx(alphas, row[1:], ".-", label=row.Index)

    alpha_min = alphas.min()
    top_coefs = coefs.loc[:, alpha_min].map(abs).sort_values().tail(n_highlight)
    for name in top_coefs.index:
        coef = coefs.loc[name, alpha_min]
        plt.text(
            alpha_min, coef, name + "   ",
            horizontalalignment="right",
            verticalalignment="center"
        )
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.grid(True)
    ax.set_xlabel("alpha")
    ax.set_ylabel("coefficient")



def alpha_for_k_features(lasso_coeff, k_features, print_alphas = True):
    """
    Finds the alpha(s) for a specified number of features and prints out the features it found for each.
    """
    explanation = dict()
    for alpha in list(lasso_coeff.columns):
        if sum(np.absolute(lasso_coeff.loc[:,alpha]) > 0) == k_features:
            if print_alphas == True:
                print("Model with alpha = ", alpha , " contains ", k_features, " features:")
                print(lasso_coeff.loc[list(lasso_coeff.loc[:,alpha] >0 ),alpha])
            explanation[alpha] = np.exp(lasso_coeff.loc[list(np.absolute(lasso_coeff.loc[:,alpha]) >0 ),alpha])
    return explanation


def plot_surv_curve(model_event_times_, surv_curve_array, linewidth=[2,4], title="", ylabel="Survival probability", plot_inverse=False):
    """
    Plots out several or a single survival curve.
    Extends x-axis to start from year 0 with probability 1 if stepfunction does not start there.
    """
    if len(surv_curve_array.shape) == 2:
        if 0 not in model_event_times_:
            model_event_times_ = np.append(0, model_event_times_)
            surv_curve_array = np.hstack([np.ones((surv_curve_array.shape[0],1)), surv_curve_array])

        if plot_inverse==True:
            surv_curve_array = 1-surv_curve_array
        for i, s in enumerate(surv_curve_array):
            plt.step(model_event_times_, s, where="post", label=str(i), linewidth=linewidth[0])
        plt.ylabel(ylabel)
        plt.xlabel("Time")
        plt.ylim(0, 1.1)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()
    elif len(surv_curve_array.shape) == 1:
        if 0 not in model_event_times_:
            model_event_times_ = np.append(0, model_event_times_)
            surv_curve_array = np.append(1,surv_curve_array)
        if plot_inverse==True:
            surv_curve_array = 1- surv_curve_array
        plt.step(model_event_times_, surv_curve_array, where="post", linewidth=linewidth[1], color='green')
        plt.ylabel(ylabel)
        plt.xlabel("Time")
        plt.ylim(0, 1.1)
        plt.title(title)
        plt.grid(True)
        plt.show()
    else:
        assert("error encountered: shape of entered array is not 1 or 2")
    pass




def add_all_interactions_to_df(d, order=None):
    k= len(list(d.columns))
    combinations= []
    if order == None:
        order = k
    for i in np.arange(2,order+1):
        combinations.extend(list(itertools.permutations( range(k)  , int(i)  )))
    permutations = []
    for i in combinations:
        permutations.append(sorted(i))
    permutations.sort()
    permutations = list(permutations for permutations,_ in itertools.groupby(permutations))
    for i in permutations:
        new_col = d.iloc[:,i].prod(axis=1)
        str = ""
        for j in list(d.iloc[:,i].columns):
            str = str + "X"
            str = str + j
        str = str[1:]
        d[str] = new_col
    return d


def train_cox_and_explain(k_features, 
                          scaled_data, 
                          y_lasso, 
                          surv_curves_pert_bb, 
                          time_stamps_bb, 
                          plot_inverse_surv_curve,
                          alpha_min_ratio=0.01,
                          plot_curve=True):

    time_stamps = time_stamps_bb
    k_features_found = False
    counter = 0
    explanation = dict()
    while k_features_found==False:
        cox_lasso = CoxnetSurvivalAnalysis(l1_ratio=1.0, alpha_min_ratio=alpha_min_ratio, fit_baseline_model = True)
        cox_lasso.fit(scaled_data, y_lasso)  
            # Investigate how many features remain for each alpha
        coefficients_lasso = pd.DataFrame(
            cox_lasso.coef_,
            index=scaled_data.columns,
            columns=np.round(cox_lasso.alphas_, 5)
        )
        
        if any(np.array(np.sum(np.absolute(coefficients_lasso)> 0))==k_features): 
            k_features_found = True
            alpha = coefficients_lasso.columns[np.array(np.sum(np.absolute(coefficients_lasso)> 0))==k_features][0]
            column_index = list(np.array(np.sum(np.absolute(coefficients_lasso)> 0))==k_features).index(True)
            print("{} features found for alpha_min_ratio {} at alpha {} in column_index {}.".format(k_features, alpha_min_ratio, alpha, column_index))
            explanation["coefficients_alpha_matrix"] = coefficients_lasso
            explanation["alpha"] = alpha
        else:
            alpha_min_ratio = alpha_min_ratio/2
            counter += 1
            if counter > 100:
                print("No suitable alpha found even after 100 iterations: consider quitting running time.")
            if counter >1000:
                print("No alpha found for {}. Trying to select {} most important features".format(k_features, k_features+1))
                k_features= k_features+1
            if counter >2000:
                print("No alpha found for {}. Trying to select {} most important features".format(k_features, k_features-2))
                k_features= k_features-2
            if counter >3000:
                assert("No suited alpha for specified number of features found.")

    # for i in range(coefficients_lasso.shape[1]):
    #     print("column ", i, " has ", np.sum(coefficients_lasso.iloc[:,i]>0), " nonzero coefficients.")

        # Get survival curve + hazard ratios for top k_features for instance of interest
    explanation["hazard_ratios"] = alpha_for_k_features(coefficients_lasso, k_features, print_alphas = False)[alpha]

        # Predict the survival curve of the instance 
    surv_curves_cox = cox_lasso.predict_survival_function(scaled_data)
    explanation["survival_curve_bb"] = surv_curves_pert_bb[0]
    explanation["timestamps_bb"] = time_stamps
    explanation["survival_curve_cox"] = surv_curves_cox[0].y
    explanation["timestamps_cox"] = surv_curves_cox[0].x


    # If there is no timestamp for 10 years the next available one is chosen for the prediction
    chosen_curve = explanation["survival_curve_bb"] # We use the predicted curve of the blackbox model 
    chosen_timestamps = explanation["timestamps_bb"]

    explanation_text = """The predictors that contributed the most to this prediction are the scores on: \n {} \nin that order of importance.
    """.format(
        list(explanation["hazard_ratios"].index[order_hazard_ratios_by_importance(explanation["hazard_ratios"])])
        )
    explanation["explanation_text"] = explanation_text

        # Calculate the fidelity (local concordance) between the RSF and penalised Cox on the training data
        # First let us look at a few examples
    #surv_utils.plot_surv_curve(cox_event_times, surv_curves_pert_cox[0], title="cox prediction")
    #surv_utils.plot_surv_curve(time_stamps, surv_curves_pert_bb[0], title= "{} prediction".format(blackbox))
    fidelity, a, b = calculate_fidelity(explanation["timestamps_cox"], explanation["timestamps_bb"], explanation["survival_curve_cox"], explanation["survival_curve_bb"])

        # To do this we calculate the mean of the absolute differences between the two curves at each time-point
    print(explanation["explanation_text"])
    print("The fidelity measured as the mean of the absolute differences between the two curves at each time-point is: ", np.round(fidelity,3))
    explanation["fidelity"] = np.round(fidelity,3)
    explanation["interpretable_model"] = cox_lasso
    if plot_curve==True:
        plot_surv_curve(chosen_timestamps, chosen_curve, title= "", plot_inverse=plot_inverse_surv_curve)
        # order the hazard ratios
    explanation["hazard_ratios"] = explanation["hazard_ratios"].sort_values(key=lambda x: -abs(x-1))
    return explanation




def convert_data_for_plot(original_df):
    original_df.columns = np.arange(1,6)
    value = []
    importance_ranking = [1,2,3,4,5]*5
    variable = np.repeat(["x1","x2","x3","x4","x5"],5)
    for i in range(len(variable)):
        value.append(sum(original_df[  importance_ranking[i] ] == variable[i]))            
    d = {"importance_ranking":importance_ranking,
            "variable":variable,
            "value":value}
    new_df = pd.DataFrame(d)
    new_df.index = np.arange(1,26)
    return new_df

