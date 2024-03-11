import pandas as pd
import numpy as np
import pickle
import surv_utils
from nonadditive_explainers import nonadditive_explainer
    # Can start from here again yb loading required data and objects
with open('/Users/abdallah/Desktop/Kings College Project/Code/Data prepation ELSA Dementia/DATA/explanations_fidelity.pkl', 'rb') as fp:
    explanations = pickle.load(fp)
with open('/Users/abdallah/Desktop/Kings College Project/Code/Data prepation ELSA Dementia/DATA/cat_feature_dict.pkl', 'rb') as fp:
    cat_feature_dict = pickle.load(fp)
data_w1_dummied = pd.read_csv("/Users/abdallah/Desktop/Kings College Project/Code/Data prepation ELSA Dementia/DATA/data_w1_dummied.csv", index_col=0)
data_w1_dummied_unstandardised = pd.read_csv("/Users/abdallah/Desktop/Kings College Project/Code/Data prepation ELSA Dementia/DATA/data_w1_dummied_unstandardised.csv", index_col=0)
selected_features = ['scorg3', 'social',
                         "cesd_sc",'n_chronic_diseases',"n_neuropsychiatric_symptoms",
                          'ethnicity_White', 'gender_Men', 'mmgsd_w2', 
                        'edqual_Degree or equiv', 'edqual_Foreign/other',
                        'edqual_NVQ/GCE O/A Level equiv', 'edqual_No qualification', 
                         'memory1', 'procspeed1', # left out 'exe1',  as it correlates strongly (0.46) with memory1 and is a similar construct
                          "age", "dem_hse_w8", "time"]
data_w1_dummied["social"] = data_w1_dummied[['scorg1', 'scorga2', 'scorg4', 'scorg5', 'scorg6', 'scorg7', 'scorg8']].sum(axis=1)

random_state = 1
    # use 100 instances for testing, the rest for training
test_indices = data_w1_dummied.sample(100, random_state=random_state).index
training_indices = [x for x in data_w1_dummied.index if x not in test_indices]
training = data_w1_dummied.loc[training_indices, selected_features]
test = data_w1_dummied.loc[test_indices, selected_features]

    # Split into predictors and targets
training_x, training_y = surv_utils.convert_time_y_to_surv_y(df = training, y = "time", event_indicator = "dem_hse_w8", only_targets = False, round_y= False) 
test_x, test_y = surv_utils.convert_time_y_to_surv_y(df = test, y = "time", event_indicator = "dem_hse_w8", only_targets = False, round_y= False)

# Case study
    # For SurvMLeX
orderingsSurvMLeX = pd.DataFrame(np.zeros((10,len(selected_features))))
orderingsSurvMLeX.columns = selected_features
for explanation in explanations["SurvMLeX"]: # Replace that with whatever tool I want and make a table
    for index, feature in enumerate(explanation[:10]):
        orderingsSurvMLeX.loc[index, feature] += 1

    # For ChoquEx
orderingsChoquEx = pd.DataFrame(np.zeros((10,len(selected_features))))
orderingsChoquEx.columns = selected_features
for explanation in explanations["SurvChoquEx"]: # Replace that with whatever tool I want and make a table
    for index, feature in enumerate(explanation[:10]):
        orderingsChoquEx.loc[index, feature] += 1

    # Get a global measure of feature importance
        #For SurvMLeX TODO repeat this but run it on the entire dataset
global_mlex = nonadditive_explainer(training_x.iloc[:1000,:], training_y[:1000], method = "SurvMLeX", alpha=1.0, finetune=True, rank_ratio=1.0, fit_intercept=False, n_shuffle_splits=5, random_state=random_state, range_alpha= (2.0 ** np.arange(-8, 7, 2)))
global_mlex
        #For SurvChoquEx
global_choquex = nonadditive_explainer(training_x.iloc[:1000,:], training_y[:1000], method = "SurvChoquEx", alpha=1.0, finetune=True, rank_ratio=1.0, fit_intercept=False, n_shuffle_splits=5, random_state=random_state, range_alpha= (2.0 ** np.arange(-8, 7, 2)))
global_choquex
    # SurvMLeX: Those are the top 5 features that are most important, 2nd most important, 3rd most importnat (make table)
for rank in range(5):
    orderingsSurvMLeX.sort_values(by=rank, axis=1).iloc[rank,-5:]
    # ChoquEx: Those are the top 5 features that are most important, 2nd most important, 3rd most importnat (make table)
for rank in range(5):
    orderingsChoquEx.sort_values(by=rank, axis=1).iloc[rank,-5:]
    # Put all those features in a Cox regression to see the nature of their relationship with dementia (look at the hazard ratios)
set_config(display="text")  # displays text representation of estimators
estimator = CoxPHSurvivalAnalysis()
estimator.fit(training_x, training_y)
pd.Series(estimator.coef_, index=training_x.columns) # Negative sign means PROTECTIVE effect. So being white, male etc is bad and ....
    # Add all interactions and try again with penalised Cox regression
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.preprocessing import OneHotEncoder
from sklearn import set_config
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
Xt = surv_utils.add_all_interactions_to_df(training_x, order=2).copy()
y = training_y.copy()
alphas = 10.0 ** np.linspace(-4, 4, 50)
coefficients = {}
cph = CoxPHSurvivalAnalysis()
for alpha in alphas:
    cph.set_params(alpha=alpha)
    cph.fit(Xt, y)
    key = round(alpha, 5)
    coefficients[key] = cph.coef_
coefficients = pd.DataFrame.from_dict(coefficients).rename_axis(index="feature", columns="alpha").set_index(Xt.columns)


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
        plt.text(alpha_min, coef, name + "   ", horizontalalignment="right", verticalalignment="center")

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.grid(True)
    ax.set_xlabel("alpha")
    ax.set_ylabel("coefficient")


plot_coefficients(coefficients, n_highlight=5)
plt.show()



    # See which 40 features remain with Lasso
k_features=40
scaled_data = Xt.copy()
y_lasso = y.copy()
alpha_min_ratio=0.01
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

    # Here the 40 most important features/interactions
coefficients_lasso.loc[coefficients_lasso.loc[:,alpha] != 0.0,alpha].sort_values(key=abs)

    # For the partial dependence plots I can use R. Will save the df for that purpose
training.to_csv("/Users/abdallah/Desktop/Kings College Project/Code/results/case_study/selected_elsa_df.csv")



    # Plot some interactions
    # Most important ones: ethnicity and processing speed (-0.29), age and processing speed (0.17), age and absence of an educational qualification (-0.15), as well as social participation and religiosity (0.07). 
import seaborn 
import matplotlib.pyplot as plt

#     #
# explainer_fidelity.train_x["ethnicity_White"].value_counts()
# explainer_fidelity.train_x["procspeed1"]

    #

data_w1_dummied_unstandardised["age_over_70"] = [1. if x >69 else 0. for x in data_w1_dummied_unstandardised["age"]]
ax = seaborn.lmplot(x="procspeed1",markers='*', scatter_kws={'alpha':0.3}, hue="age_over_70", y="time", y_jitter=0.3, x_jitter=0.3,data=data_w1_dummied_unstandardised)
ax.set(xlabel='Processing Speed', ylabel='Time to Dementia')
ax._legend.set_title("Age")
new_labels = ['< 70', '> 70']
for t, l in zip(ax._legend.texts, new_labels):
    t.set_text(l)

# Access the legend object and set frameon to True
ax._legend.set_frame_on(True)
ax._legend.set_bbox_to_anchor((1, 0.3))

plt.show()


#     #
# explainer_fidelity.train_x["age"]
# explainer_fidelity.train_x["edqual_No qualification"].value_counts()
# seaborn.lmplot(x="age", 
#                hue="edqual_No qualification", 
#                y="time", 
#                markers='*', 
#                data=training) 
# plt.show()

    #
training_x["scorg3"].value_counts()
training_x["social"].value_counts()




data_w1_dummied_unstandardised["social"] = data_w1_dummied["social"]
ax = seaborn.lmplot(x="social",markers='*', scatter_kws={'alpha':0.3},  hue="scorg3", y="time", y_jitter=0.3, x_jitter=0.3, data=data_w1_dummied_unstandardised)

ax.set(xlabel='Social Participation', ylabel='Time to Dementia')
ax._legend.set_title("Part of Religious Group")
new_labels = ['No', 'Yes']
for t, l in zip(ax._legend.texts, new_labels):
    t.set_text(l)

# Access the legend object and set frameon to True
ax._legend.set_frame_on(True)
ax._legend.set_bbox_to_anchor((1.2, 0.3))

plt.show()
np.mean(data_w1_dummied_unstandardised.loc[data_w1_dummied_unstandardised["scorg3"]==0,"time"])



