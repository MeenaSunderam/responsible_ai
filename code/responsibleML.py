import string
import numpy as np
import as pd
from matplotlib import pyplot as plt

import json
from enum import Enum
import logging

import fairlens as fl
import shap
from codecarbon import EmissionsTracker
from captum.attr import IntegratedGradients

class ProblemType(Enum):
    """Type of problem deduced from the label values"""

    BINARY = "binary_classification"
    REGRESSION = "regression"
    MULTICLASS = "multiclass_classification"
    FORECASTING = "forecasting"
    RECOMMENDATION = "recommendation"
    OTHER = "other"

class ModelFramework(Enum):
    """Type of Framework used to train the model"""

    SKLEARN = "sklearn"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    OTHER = "other"
    
class DataType(Enum):
    """Type of data used to train the model"""

    TABULAR = "tabular"
    TEXT = "nlp"
    IMAGE = "vision"
    TIMESERIES = "time series"
    
class ResponsibleMetrics(Enum):
    
    BIAS = "bias"
    EMISSIONS = "emissions"
    CLASS_IMBALANCE = "class_imbalance"
    INTERPRETABILITY = "interpretability"
    
class Emissions_Level(Enum):
    """Level of Emissions"""

    LOW = 500
    MEDIUM = 10000

class responsible_model:
    """Type of Framework used to train the model"""
    
    __model_name = None
    __framework = ModelFramework.SKLEARN
    __ml_problem = ProblemType.BINARY
    __datatype = DataType.TABULAR
    
    __emissions = 0.0
    __class_balance  = 0.0
    __interpretability = 0.0
    
    __emissions_index = 0
    __class_balance_index = 0
    __interpretability_index = 0
    
    __model_index = 0.0
    __model_accuracy = 0.0
    
    index_weightage = "EQUAL"
    
    ### EmissionsTracker ###
    __tracker = None
    
    def __init__(self, model_name:string, ml_problem: ProblemType, framework: ModelFramework):
        
        # General Model information
        self.__model_name = model_name
        self.__framework = framework
        self.__ml_problem = ml_problem
        
        # Responsible Model Metrics
        self.__emissions = 0.0
        self.__class_balance = 0.0
        self.__interpretability = 0.0
        
        # Responsible Index
        self.__emissions_index = 0
        self.__class_balance_index = 0
        self.__interpretability_index = 0
        
        # Overall Responsible Index
        self.__model_index = 0.0
        self.__model_accuracy = 0.0 
                
    def get_model_name(self)->string:
        return self.__model_name
    
    def get_framework(self)->ModelFramework:
        return self.__framework
    
    def get_model_type(self)->ProblemType:
        return self.__ml_problem
    
    def get_emissions(self)->float:
        return self.__emissions
    
    def get_class_balance(self)->float:
        return self.__class_balance
    
    def get_interpretability(self)->float:
        return self.__interpretability

    def get_emissions_index(self)->float:
        if self.__emissions_index == 0 :
            self.__calculate_emissions_index()
            
        return self.__emissions_index
    
    def get_interpretability_index(self)->float:
        if self.__interpretability_index == 0:
            self.__calculate_interpretability_index()
        
        return self.__interpretability_index
    
    def get_class_balance_index(self)->float:
        if self.__class_balance_index == 0:
            self.__calculate_class_balance_index()
            
        return self.__class_balance_index
    
    def set_model_name(self, model_name):
        self.__model_name = model_name
        
    def set_framework(self, framework):
        self.__framework = framework
        
    def set_data_type(self, data_type: DataType):
        self.__datatype = data_type
                
    def set_emissions(self, emissions):
        self.__emissions = emissions
        
    def set_class_balance(self, class_balance):
        self.__class_balance = class_balance
    
    def set_interpretability(self, interpretability):
        self.__interpretability = interpretability
        
    def set_index_weightage(self, index_weightage):
        self.index_weightage = index_weightage
        
    def set_model_accuracy(self, accuracy):
        self.__model_accuracy = accuracy
        
    def get_model_info(self):
        
        value = json.dumps({"model name": self.__model_name,
                    "framework": self.__framework.value,
                    "ml problem": self.__ml_problem.value,
                    "data type": self.__datatype.value,
                    "model_accuracy": self.__model_accuracy,
                    "emissions": self.__emissions,
                    "class_balance": self.__class_balance,
                    "interpretability": self.__interpretability,
                    "class balance Index": self.__class_balance_index,
                    "interpretability index": self.__interpretability_index,
                    "emission index": self.__emissions_index,
                    "model_rai_index": self.__model_index})
        
        return value
    
    ### ---------- Emissions Index ---------- ###
    
    def start_emissions_tracker(self):
        self.__tracker = EmissionsTracker()
        self.__tracker.start()
    
    def stop_emissions_tracker(self):
        self.__emissions : float = self.__tracker.stop()
        self.__calculate_model_index()
        
    def __calculate_emissions_index(self):
        if self.__emissions <= 500:
            self.__emissions_index = 3
        elif self.__emissions > 500 and self.emissions <= 10000:
            self.__emissions_index = 2
        else:
            self.__emissions_index = 1
        
    ### ---------- Class Balance Index ---------- ###
    
    def calculate_class_balance(self, df_label: pd.DataFrame):
        
        # Get the number of classes & samples 
        label_classes = df_label.value_counts(ascending=True)
        
        optimal_distribution = 1 / label_classes.count()
        min_class_distribution = label_classes.values[0]/label_classes.sum()
        
        #calcualte the Class Balance
        self.__class_balance = min_class_distribution/optimal_distribution
        
        self.__calculate_model_index()
            
    def __calculate_class_balance_index(self):
        if self.__class_balance >= 0.4:
            self.__class_balance_index = 3
        elif self.__class_balance > 0.2 and self.__class_balance < 0.4:
            self.__class_balance_index = 2
        else:
            self.__class_balance_index = 1
    
     ### ---------- Bias Index ---------- ###     
    
    def calculate_bias(self, df, label:string, sensitive_attributes:list):
        
        #if no sensitive attributes are provided, identify them automatically
        if len(sensitive_attributes) == 0:
               sensitive_attributes =  fl.FairnessScorer(df, label).sensitive_attrs
        
        #get the fairness score for the sensitive attributes
        fscorer = fl.FairnessScorer(df, "target", sensitive_attributes)
        
    def __calculate_bias_index(self):
        return
    
    ### ---------- Interpretability Index ---------- ###        
    
    def calculate_interpretability(self, model_type, model, df_x):
        
        # Use Model co-eff if just the model is specified
        if df_x is None:
            if model_type == 'linear':
                importance = model.coef_[0]
            elif model_type == 'treebased'
                importance = model.feature_importances_
            
            vals = np.abs(importance)
            sorted_vals = np.sort(vals)
            top3 = sorted_vals[-3:].sum()
            total = sorted_vals.sum()

            self.__interpretability = top3 / total

            self.__calculate_model_index()
        
        # Use SHAP otherwise
        else:
            # Explain model predictions using shap library:
            shape_values_df = None

            if model_type == 'linear':
                explainer = shap.LinearExplainer(model, df_x, feature_dependence="interventional")
                shap_values = explainer.shap_values(df_x)
                shape_values_df = pd.DataFrame(shap_values, columns=df_x.columns)     

            elif model_type == 'treebased':        
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(df_x)
                shape_values_df = pd.DataFrame(shap_values[1], columns=df_x.columns)

            vals = np.abs(shape_values_df.values).mean(0)   
            sorted_vals = np.sort(vals, axis=0)        
            top3 = sorted_vals[-3:].sum()
            total = sorted_vals.sum()

            self.__interpretability = top3 / total

            self.__calculate_model_index()
    
    def __calculate_interpretability_index(self):
        
        if self.__interpretability >= 0.6:
            self.__interpretability_index = 3
        elif self.__interpretability > 0.4 and self.__interpretability < 0.6:
            self.__interpretability_index = 2
        else:
            self.__interpretability_index = 1
    
    ### ---------- Responsible Model Index ---------- ###                
    
    def __calculate_model_index(self):
        self.__calculate_emissions_index()
        self.__calculate_class_balance_index()
        self.__calculate_interpretability_index()
        
        if self.index_weightage == "EQUAL":
            self.__model_index = (self.__emissions_index + self.__class_balance_index + self.__interpretability_index) / 3
        
        return self.__model_index

#############################################################################################
##################################### PyTorch Model #########################################
#############################################################################################

class pytorch_model(responsible_model):
    
    def __init__(self, model_name):
        super().__init__(model_name)
        super().set_framework(ModelFramework.PYTORCH)
    
    ### ---------- Overwrite Interpretability Index ---------- ###        
    def calculate_interpretability(self, input_tensor, model,target_class):

        ig = IntegratedGradients(model)
        input_tensor.requires_grad_()
        attr, delta = ig.attribute(input_tensor,target=target_class, return_convergence_delta=True)
        attr = attr.detach().numpy()
        importance = np.mean(attr, axis=0)
        
        importance = np.abs(importance)        
        importance[::-1].sort()
        
        total_weightage = np.sum(importance)
        key_features_weightage = importance[0] + importance[1] + importance[2]
        
        super().set_interpretability = key_features_weightage / total_weightage
        
#############################################################################################
##################################### TensorFlow Model  #####################################
#############################################################################################

class tensorflow_model(responsible_model):
    
    def __init__(self, model_name):
        super().__init__(model_name)
        super().set_framework(ModelFramework.TENSORFLOW)
    
    ### ---------- Overwrite Interpretability Index ---------- ###        
    def calculate_interpretability(self, input_tensor, model,target_class):

        ig = IntegratedGradients(model)
        input_tensor.requires_grad_()
        attr, delta = ig.attribute(input_tensor,target=target_class, return_convergence_delta=True)
        attr = attr.detach().numpy()
        importance = np.mean(attr, axis=0)
        
        importance = np.abs(importance)        
        importance[::-1].sort()
        
        total_weightage = np.sum(importance)
        key_features_weightage = importance[0] + importance[1] + importance[2]
        
        super().set_interpretability = key_features_weightage / total_weightage

#############################################################################################
##################################### RAI Models        #####################################
#############################################################################################

class rai_models:
    model_list = []
    
    def __init__(self):
        self.model_list = []
        
    def add_model(self, model):
        self.model_list.append(model)
        
    def remove_model(self, modelname):
        self.model_list.remove(modelname)
        
    def list_models(self):
        model_json = ""
        for model in self.model_list:
            model_json += model.get_model_info() 
            if model != self.model_list[-1]:
                model_json += ","
                                
            model_json += "\n"
            
        model_json = "[" + model_json + "]"
        
        return model_json
    
    def get_model(self, modelname):
        for model in self.model_list:
            if model.get_model_name() == modelname:
                return model
        return "Model information NOT Found"
    
    def rank_models(self, rank_by = "rai_index"):
        sorted_json = ""
        
        if rank_by == "rai_index":
            sorted_models = sorted(self.model_list, key=lambda x: x.get_model_index(), reverse=True)
        elif rank_by == "emissions":
            sorted_models = sorted(self.model_list, key=lambda x: x.get_emissions_index(), reverse=True)
        elif rank_by == "bias":
            sorted_models = sorted(self.model_list, key=lambda x: x.get_class_balance_index(), reverse=True)
        elif rank_by == "interpretability":
            sorted_models = sorted(self.model_list, key=lambda x: x.get_interpretability_index(), reverse=True)
            
        for model in sorted_models:
            sorted_json += model.model_rai_components()
            if(model != sorted_models[-1]):
                sorted_json += ","
            sorted_json += "\n"
            
        sorted_json = "[" + sorted_json + "]"
        return sorted_json