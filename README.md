# Diagnostic Prediction: Let’s make age just a number
Final Project_WBS Coading School - Data Science Bootcamp

## Abstract
This was my final project during my WBS Coding School Data Science Bootcamp. The project concerns the Kaggle Competition "ICR - Identifying Age-Related Conditions" (https://www.kaggle.com/competitions/icr-identify-age-related-conditions/overview). The goal is to assess the ability of machine learning to find patterns that could help discover the relationships between key characteristics and potential patient conditions, ultimately improving healthcare outcomes.

At the moment of the final project presentation the competition was still ongoing: I was at the place 1.288 from 6.325 teams (7.106 participants) with a log loss of 0.06. The leading team achieved a log loss of 0.00. The final results will be based on bigger/different test set, so the final standings may be different.

## Context
Aging comes with a variety of health issues and is a risk factor for numerous diseases and complications. The field of bioinformatics aims to research interventions that can slow and reverse biological aging and prevent age-related ailments. In this competition, participants will work with health characteristic data to develop a predictive model that can improve existing methods for predicting medical conditions. Prize for this competition is 60.000 dolars.

## Goal of the Competition
The goal of this competition is to predict if a person has any of three medical conditions. The participants are required to predict whether a subject has been diagnosed with one or more of the three medical conditions (Class 1) or none of the three medical conditions (Class 0). This prediction will be based on measurements of health characteristics provided in the dataset.

## Dataset Description
The competition data consists of 56 anonymized health characteristics linked to three age-related conditions. The training set containing unique identifiers for each observation, fifty-six anonymized health characteristics (numeric), and a binary target variable (1 for diagnosed, 0 for not diagnosed). The test set for which the participants must predict the probability that a subject belongs to each of the two classes. Supplemental metadata available only for the training set, containing identifiers for the type of age-related condition, if present, and three experimental characteristics.
In the Notebook "Dataset_Visualization" (main branch) a comprehensive analysis of the dataset is presented.

## Evaluation Metric
Submissions are evaluated using a balanced logarithmic loss, where each class is roughly equally important for the final score. The formula for the evaluation metric is provided in the competition description.

## Scoring
This is a Code Competition, where the actual test set is hidden. Participants are provided with sample data in the correct format for authoring solutions. When a submission is scored, the sample test data will be replaced with the full test set, consisting of approximately 400 rows. The leaderboard is calculated with approximately 42% of the test data, and the final results will be based on the remaining 58%. Participants may submit a maximum of 1 entry per day, and for the final competition, only 2 submissions will count towards the final leaderboard score.

## Description of Notebooks Selected
During the competition, I had successfully submitted 10 notebooks. For the final competition, two submissions will count towards the final leaderboard score. At the moment of the final project presentation the competition was still ongoing and the final results could be different. However I selected two differet machine learning model with the best score:

#### Best_Result_tabPFN-XGBoost-stack_score=0.06
- Random undersampling for the balance the "Class" distribution ("train" dataframe).
- Two cross-validation strategies using K-Fold in order to get a more reliable estimate of the model's performance on unseen data and avoid overfitting.
- Ensemble model that combines the predictions of two classifiers: XGBClassifier from XGBoost and TabPFNClassifier from the TabNet framework.
- Training function was performed by training and evaluation of a given model using nested cross-validation.
- Random oversampling with respect to the "greeks" dataframe.
- Final training on the balanced datase that assess the model's performance and returns the best trained model.
- Best trained model was used to make predictions on test data.

#### Other_models_XGBoost_with_RandomsearchCV_score=0.26
- Apply RandomSearchCV to find the best parameters.
- XGBClassifier from XGBoost with the following hyperparameters:
                      subsample= 0.9, 
                      min_child_weight = 2, 
                      max_depth= 2, 
                      n_estimators= 50,
                      gamma= 1, 
                      colsample_bytree= 0.8,
                      random_state= state
- Performs k-fold cross-validation using KFold and cross_val_score functions.

## Other models
Besides the Notebook Selected for submission are Classifiers were used: DesicionTreeClassifier, RandomForestClassifier, KNeighborsClassifier, LightGBM Classifier. In the main branch there are notebooks where this classifiers were tested. One of the notebook ("Other-models-Classifiers-stacking") also present stacking of three classifiers using GridSearchCV for best parameters selection. However, there the overfitting was a big problem when stacking was performed. Even after cross-validation the results showed low accuracy. This notebook was not considered for submission, but show how stacking of different models can be performed in a simple way. In the notebook "Other_models_Neural-Network_tensorflow" I builded a Neural Network sequential model using the Keras 'Sequential' API and optimization of architecture using features as 'kernel_regularizer', 'Dropout', number of hiddedn layers as activation function for the binary classification problem.

## Tools
●	Pandas
●	Seaborn
●	Scikit-Learn
●	Tensorflow 
