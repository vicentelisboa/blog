---
date: 2023-03-27T10:58:08-04:00
description: "By Vicente Lisboa"
featured_image: "/images/background_4.jpg"
tags: ["Decision trees", "Neural networks"]
title: "Probability of death and length of stay using MIMIC dataset"
---

## <b style="font-size: 32px"> Introduction</b>

The following project has two objectives:

+ Predict the probability of death of a patient that is entering an ICU (Intensive Care Unit) using K-Nearest Neighbors (KNN) and Support Vector Machine (SVM) models.
+ Predict the length of stay (in days) of a patient that is entering an ICU using decision tree, ensembles and neural networks models.


## <b style="font-size: 32px"> Dataset description</b>

The dataset comes from MIMIC project (https://mimic.physionet.org/). MIMIC-III (Medical Information Mart for Intensive Care III) is a large, freely-available database comprising deidentified health-related data associated with over forty thousand patients who stayed in critical care units of the Beth Israel Deaconess Medical Center between 2001 and 2012.

## <b style="font-size: 32px"> Feature creation</b>

+ The aim was to define features that could help in predicting the probability of death and the length of stay of a patient that is entering an ICU
+ The following features were created:
  + Age
  + Comorbilities: Number of comorbilities of the patient
  + Repeated visits to the ICU: Number of times that a patient visited the ICU
+ Categorical variables: Redefine categories
  + Ethnicity
  + Religion

+ Imputation:
  + KNN imputation
  + One hot encoding

## <b style="font-size: 32px"> Models</b>

## <b style="font-size: 32px"> Probability of death</b>

+ To predict the probability of death, I used a grid search to find the better parameters of the following models:
  + K-Nearest Neighbors model
  + Support Vector Machine model

## <b style="font-size: 32px"> K-Nearest Neighbors model</b>

+ Performance of the model:

{{< figure src="/images/KNN_train_score.png" title="Grid search table for KNN" >}}

Train set metrics:

Metric | Score |
--- | ---
Accuracy  |	0.891 |
Precision  |	0.986 |  
Recall  |	0.030 |
F1 Score |	0.058 |


## <b style="font-size: 32px"> Support Vector Machine model</b>

+ Performance of the model:

{{< figure src="/images/SVM_train_score.png" title="Grid search table for KNN" >}}

Train set metrics:

Metric | Score |
--- | ---
Accuracy  |	0.828 |
Precision  |	0.380 |  
Recall  |	0.836 |
F1 Score |	0.522 |


## <b style="font-size: 32px"> Decisions tree, ensembles and neural networks models</b>

+ To predict the length of stay, I used a grid search to find the better parameters of the following models:

  + XGB Regressor
  + Decision Tres Regressor
  + Random Forest Regressor
  + Ada Boost Regressor
  + Gradient Boosting Regressor


+ Model comparison

{{< figure src="/images/dt_model_comparison.png" title="Decisions Tree Model performance comparative" >}}


+ Ensembles Models
  + I also implement ensembles models
  + For that firstly I investigated Metalearner Candidates and then fit three models to train to choose the best metalearner for stacking ensemble
  + The metalearner introduced were:
    + Decision Tree Regressor
    + Random Forest Regressor
    + Gradient Boosting Regressor.


+ General results:

Model | Mean Score Error |
--- | ---
XGB_Regressor  |	21.58 |
GradientBoosting_Regressor  |	21.80 |  
StackingRandomForest  |	21.81 |
StackingDecisionTrees |	22.29 |
StackingGradientBoosting  |	22.33 |  
RandomForest_Regressor  |	23.41 |
DecisionTree_Regressor  |	24.08 |
AdaBoost_Regressor |	24.87 |

The chosen model was the XGB_Regressor according to the mean score error criteria


+ Neural Networks:
  + I run an initial neural network model that consist in:
    + Sequential constructor takes an array of keras Layers
    + Define a range of number of epoch, that are the number of times the learning algorithm will work through the whole dataset)
    + The performance of the model according to the MSE by the iteration of epchs


    {{< figure src="/images/neural network.png" title="Performance of the neural network model" >}}


+ Model interpretability
  + Here I present the variables that have more impact in the model
  + It's possible to see that the number of previous comorbilities it's the feature with more impact in the chosen model

{{< figure src="/images/impact in the model.png" title="Model interpretability" >}}

 ## <b style="font-size: 32px"> Codes</b>

You can read my work with more detail in the attached github.

**GitHub Repository:** https://github.com/vicentelisboa/probability_of_death_and_length_of_stay_using_MIMIC_dataset-.git
