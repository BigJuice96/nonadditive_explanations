import surv_utils
from nonadditive_explainers import nonadditive_explainer
import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance
from survshap import SurvivalModelExplainer, PredictSurvSHAP



class ExplainSurvival:
    def __init__(self, 
                 model, # the model whose output we want to explain 
                 x, # DUMMIED data that was used to train the blackbox model
                 y, # the outcome data in a format suitable for sklearn survival models  
                 categorical_features, # List of strings of the feature labels corresponding to categorical features (noncategorical features are discretised)
                 random_state = None, # random state for intilisation of random functions
                 ):
            # Save important objects as attributes
        y.dtype = [('Status', '?'), ('Survival_in_days', '<f8')] # Some of Majid's code throws warnings if dtype differs
        self.random_state = random_state
        self.model = model 
        self.categorical_features = categorical_features
        # self.time_stamps = model.unique_times_ 
        self.time_stamps = model.predict_survival_function(x.iloc[:10,:])[0].x # also works if model.unique_times_ is deprecated
        self.x, self.y = x, y

    def global_feature_importance(self, method = "pfi", n_repeats=15):
        """
        Gives the permutation feature importance (global) for the given model.
        View  sklearn.inspection.permutation_importance for more documentation.
        """
        if method == "pfi":
            pfi = permutation_importance(self.model, self.x, self.y, n_repeats=n_repeats, random_state=20)
            pfi_rankings = pd.DataFrame(
                {k: pfi[k]
                    for k in (
                        "importances_mean",
                        "importances_std",
                    )
                },
                index=self.x.columns,
            ).sort_values(by="importances_mean", ascending=False)
            pfi_rankings["feature"] = pfi_rankings.index
    #            pfi_rankings = pfi_rankings.drop(["APOEe4_No","APOEe4_Apoe present"],axis=0).iloc[:15,:]
            pfi_rankings.index = range(len(pfi_rankings.index))
            pfi_rankings.columns = ["feature_importance", "importances_std", "feature"]
            self.pfi_rankings = pfi_rankings
            print("Predictors that can be readily assessed for selection (ordered by importance): \n\n", pfi_rankings["feature"])
            print("Please select the predictors that you would like to assess create a new ExplainSurvival object by passing the feature labels as a list to the feature_selection argument.")
            return pfi_rankings
        elif method == "SurvMLeX":
            global_importance = nonadditive_explainer(self.x, self.y, self.categorical_features, method = "SurvMLeX", alpha=0.1, finetune=True, rank_ratio=1.0, fit_intercept=False, n_shuffle_splits=5, random_state=1, range_alpha= (2.0 ** np.arange(-8, 7, 2)))
            return global_importance
        elif method == "SurvChoquEx":
            global_importance = nonadditive_explainer(self.x, self.y, self.categorical_features, method = "SurvChoquEx", alpha=0.1, finetune=True, rank_ratio=1.0, fit_intercept=False, n_shuffle_splits=5, random_state=1, range_alpha= (2.0 ** np.arange(-8, 7, 2)))
            return global_importance
       

    def explain_instance(self, 
                        instance, # Pass instance as a np.array or pd.series                         
                        method = "SurvMLeX", # Explanation method: either "SurvMLeX", "SurvChoquEx", "SurvLIME" or "SurvSHAP"
                        threshold_aggregation = 0.5, # Which threshold to use when aggregating survival curves to a time-to-event estimate using the percentile method (if it is not finetuned)
                        find_best_cutoff = True, # Finds optimal value for threshold_aggregation argument
                        n_perturbed=1000, # Size of the perturbed dataset
                        kernel_width = None, # Which kernel width to use to generate the perturbed samples
                        finetune_explainer=True, # Whether to finetune expainer (only for SurvMLeX, SurvChoquEx and SurvSHAP)
                        only_shapley_values=False, # It wont use identified features to fit a Cox regression model with interactions but give the shapley values (or hazard ratios for SurvLIME) right away
                        k_features=3, # The number of features to retain if a penalised Cox regression is fit (only if only_shapley_values=False)
                        plot_curve=True, # Whether explainer should also plot survival curve (only if only_shapley_values=False)
                        alpha_min_ratio=0.01, # alpha_min_ratio argument for the penalised Cox regression (only if only_shapley_values=False)
                        alpha=1.0, # Alpha parameter of SurvMLeX or SurvChoquEx
                        survshap_calculation_method="sampling", # for survshap method: can be either "kernel", "sampling", "treeshap" or "shap_kernel"
                        order_of_interactions = None # order of interactions to consider. If None, it will consider all combinations of interactions.
                        ):
        explanation = {"hazard_ratios":np.nan, "survival_curve_bb": np.nan, "timestamps_bb":np.nan, 
                                        "survival_curve_cox":np.nan, "timestamps_cox":np.nan,"explanation_text":np.nan,
                                        "fidelity":np.nan, "interpretable_model":np.nan, "shapley_values":np.nan
                                        } # Initialise an empty explanation object:
                                            # hazard_ratios: coefficient of the local surrogate (Cox) model
                                            # survival_curve_bb and timestamps_bb: survival curve and timestamps of blackbox model prediction                             
                                            # survival_curve_cox and timestamps_cox: survival curve and timestamps of the local surrogate (Cox) model prediction 
                                            # explanation_text: string to print for a free text format explanation 
                                            # fidelity: fidelity (mean of the absolute differences between the survival functions of blackbox and surrogate models at each time-point)
                                            # interpretable_model: the local surrogate (Cox) model
                                            # shapley_values: ordered list of shapley values (for SurvMLeX, SurvChoquEx and SurvSHAP)


            # Generate perturbed dataset in the neighbourhood of the instance to be explained
        self.perturbed_data = surv_utils.generate_neighbourhood(
            training_features = self.x,
            data_row = instance,
            random_state = self.random_state,
            num_samples=n_perturbed,
            kernel_width = kernel_width
            )[0]
        
            # Get the time-to-event predictions for the neighbours using from the bb model 
        if find_best_cutoff ==True: 
            threshold_aggregation = surv_utils.find_optimal_threshold(self.x, self.y, self.model, self.model.unique_times_)
        self.perturbed_data_targets = surv_utils.create_survival_targets(self.model, self.perturbed_data, self.model.unique_times_, self.y, threshold = threshold_aggregation)
            # Feature selection with SVM 
        self.perturbed_survival_curves = self.model.predict_survival_function(self.perturbed_data, return_array=True)
        self.perturbed_data = pd.DataFrame(self.perturbed_data, columns=self.x.columns)
        
        if method == "SurvMLeX":
            shapley_values = nonadditive_explainer(self.perturbed_data, self.perturbed_data_targets, self.categorical_features, method = "SurvMLeX", alpha=alpha, finetune=finetune_explainer, rank_ratio=1.0, fit_intercept=False, n_shuffle_splits=5, random_state=self.random_state, range_alpha= (2.0 ** np.arange(-8, 7, 2)))
        elif method == "SurvChoquEx":
            shapley_values = nonadditive_explainer(self.perturbed_data, self.perturbed_data_targets, self.categorical_features, method = "SurvChoquEx", alpha=alpha, finetune=finetune_explainer, rank_ratio=1.0, fit_intercept=False, n_shuffle_splits=5, random_state=self.random_state, range_alpha= (2.0 ** np.arange(-8, 7, 2)))
        elif method == "SurvSHAP":
                # create explainer
            survshap_explainer = SurvivalModelExplainer(model = self.model, data = self.x, y = self.y)
                # compute SHAP values for a single instance        
            survshap = PredictSurvSHAP(calculation_method = survshap_calculation_method)
            survshap.fit(explainer = survshap_explainer, new_observation = instance)
                # Sort and save Shapley values
            shapley_values = survshap.result[["aggregated_change","variable_name"]].sort_values(by="aggregated_change", key=lambda x: -abs(x))
            shapley_values.columns = ["shapley_values","feature_names"]
            if plot_curve == True: survshap.plot()
        elif method == "SurvLIME":
            try:
                explanation = surv_utils.train_cox_and_explain(k_features ,self.perturbed_data, self.perturbed_data_targets, self.model.predict_survival_function(self.perturbed_data, return_array=True), self.time_stamps, 
                                        False, alpha_min_ratio=alpha_min_ratio, plot_curve=False)
            except: 
                explanation = {"hazard_ratios":np.nan, "survival_curve_bb": np.nan, "timestamps_bb":np.nan, 
                                            "survival_curve_cox":np.nan, "timestamps_cox":np.nan,"explanation_text":np.nan,
                                            "fidelity":np.nan, "interpretable_model":np.nan, "shapley_values":np.nan }
                print("SurvLIME could not converge: try again")
                return explanation
            shapley_values = None
            # Save important objects
        else:
            print("No valid method specified: choose between SurvMLeX, SurvChoquEx, SurvLIME or SurvSHAP.")
            return
        self.instance, explanation["shapley_values"]  = instance, shapley_values

        if only_shapley_values==True: return explanation
            # Retain only the k most important features
        if method != "SurvLIME":
            self.perturbed_data = self.perturbed_data[list(shapley_values["feature_names"][:k_features])]
        else: 
            self.perturbed_data = self.perturbed_data[list(explanation["hazard_ratios"].index[:k_features])]
            # Add interactions 
        self.perturbed_data = surv_utils.add_all_interactions_to_df(self.perturbed_data, order=order_of_interactions)
            # Fit a penalised cox model 
#        return len(list(self.perturbed_data.columns)), self.perturbed_data,  self.perturbed_data_targets,self.perturbed_survival_curves, self.time_stamps
    

        try:
            explanation = surv_utils.train_cox_and_explain(len(list(self.perturbed_data.columns)) , # I.e., retain all features in scaled data (which is k_features + interactions)
                                    self.perturbed_data, 
                                    self.perturbed_data_targets, 
                                    self.perturbed_survival_curves, 
                                    self.time_stamps,
                                    False,
                                    alpha_min_ratio=alpha_min_ratio,
                                    plot_curve=plot_curve
                                    )
        except: 
            print("Local surrogate model (penalised Cox) could not converge to a solution (2).")
            explanation = {"hazard_ratios":np.nan, "survival_curve_bb": np.nan, "timestamps_bb":np.nan, 
                                        "survival_curve_cox":np.nan, "timestamps_cox":np.nan,"explanation_text":np.nan,
                                        "fidelity":np.nan, "interpretable_model":np.nan, "shapley_values":np.nan    }
            # Update scaled data and explanation objects in memory
        self.perturbed_data, explanation["shapley_values"] = self.perturbed_data, shapley_values
        return explanation
    




