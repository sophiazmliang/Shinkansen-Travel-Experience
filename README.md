# **Shinkansen Travel Experience**

## **Context:**

This problem statement is based on the Shinkansen Bullet Train in Japan, and passengers’ experience with that mode of travel. This machine-learning exercise aims to determine the relative importance of each parameter with regard to their contribution to the passengers’ overall travel experience. The dataset contains a random sample of individuals who traveled on this train. The on-time performance of the trains along with passenger information is published in a file named ‘Traveldata_train.csv’.  These passengers were later asked to provide their feedback on various parameters related to the travel along with their overall experience. These collected details are made available in the survey report labeled ‘**Surveydata_train.csv**’.

In the survey, each passenger was explicitly asked whether they were satisfied with their overall travel experience or not, and that is captured in the data of the survey report under the variable labeled ‘**Overall_Experience**’.


## **Objective:**

 The objective of this problem is to understand which parameters play an important role in swaying passenger feedback towards a positive scale. You are provided test data containing the travel data and the survey data of passengers. Both the test data and the train data are collected at the same time and belong to the same population.

## **Key Takeaways**

Hackathon

* The goal is to predict whether a passenger was satisfied or not considering his/her overall experience of traveling on the Shinkansen Bullet Train.
* Since we won't know our models' accuracy until they're submitted, the hackathon involves iterative experimentation with different combinations of algorithms and data preprocessing methods.
* The highest accuracy reached 0.9579799 which earned rank #3 in the 4-day Hackathon of this batch.


The following algorithms / methods have been invovled.

**Data Imputation**
* Mean / Mode / Constant Imputation
* Multivariate Imputation by Chained Equations (MICE) - uses the IterativeImputer with BayesianRidge estimator. This method is an advanced imputation technique that models each feature with missing values as a function of other features in a round-robin fashion, iterating over the features until the imputed values converge.

**Feature Encoding**
* Cumulative Distribution-Based Scaling - Convert categorical values to numerical values based on their cumulative distribution

**Feature Selection**
* Feature Importances - Extract directly from the trained models, for example, tree-based models.
* Chi-Squared Test - Uses statistical tests to select the most relevant features based on their relationship with the target variable
* Permutation Importance - Measures the importance of features by evaluating the effect of randomly shuffling the feature values on the model's performance

**Algorithms**

Linear Models
* Logistic Regression
* Stochastic Gradient Descent
* Ridge Classifier

Nearest Neighbors
* K-Nearest Neighbors

Decision Trees and Ensemble Methods
* Decision Tree
* Random Forest
* AdaBoost
* Gradient Boosting
* HistGradientBoostingClassifier

Support Vector Machines
* Support Vector Machine

Naive Bayes
* Gaussian Naive Bayes
* Multinomial Naive Bayes
* Bernoulli Naive Bayes

Neural Networks
* MLP Classifier

Discriminant Analysis
* Quadratic Discriminant Analysis

Gradient Boosting Variants
* XGBoost
* LightGBM
* CatBoost

**Hybrid Model**

* Majority Voting
* Weighted Voting



