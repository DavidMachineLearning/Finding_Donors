# # Machine Learning Engineer Nanodegree
# ## Supervised Learning
# ## Project: Finding Donors for *CharityML*

# ## Getting Started
# In this project, you will employ several supervised algorithms of your choice
# to accurately model individuals' income using data collected from the 1994 U.S. Census.
# You will then choose the best candidate algorithm from preliminary results
# and further optimize this algorithm to best model the data.
# Your goal with this implementation is to construct a model that accurately predicts
# whether an individual makes more than $50,000.
# This sort of task can arise in a non-profit setting, where organizations survive on donations.
# Understanding an individual's income can help a non-profit better understand
# how large of a donation to request, or whether or not they should reach out to begin with.
# While it can be difficult to determine an individual's general income bracket directly from public sources,
# we can (as we will see) infer this value from other publically available features.
# 
# The dataset for this project originates from the [UCI Machine Learning Repository]
# (https://archive.ics.uci.edu/ml/datasets/Census+Income).
# The datset was donated by Ron Kohavi and Barry Becker,
# after being published in the article _
# "Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid".
# You can find the article by Ron Kohavi [online](https://www.aaai.org/Papers/KDD/1996/KDD96-033.pdf).
# The data we investigate here consists of small changes to the original dataset,
# such as removing the "fnlwgt" feature and records with missing or ill-formatted entries.

# Exploring the Data
# "Income" will be our target label (whether an individual makes more than, or at most, $50,000 annually).
# All other columns are features about each individual in the census database.


import platform
print("Python version: ", platform.python_version())


import numpy as np
import pandas as pd
from time import time
from IPython.display import display
from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import fbeta_score, accuracy_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
import visuals as vs  # Import supplementary visualization code visuals.py


# Load the Census dataset
data = pd.read_csv("census.csv")
display(data.head(n=1))


# Data Exploration
n_records = data.shape[0]
n_greater_50k = len([income for income in data["income"] if income.lower() == ">50k"])
n_at_most_50k = len([income for income in data["income"] if income.lower() == "<=50k"])
greater_percent = n_greater_50k * 100 / n_records

print("Total number of records: {}".format(n_records))
print("Individuals making more than $50,000: {}".format(n_greater_50k))
print("Individuals making at most $50,000: {}".format(n_at_most_50k))
print("Percentage of individuals making more than $50,000: {:.3f}%".format(greater_percent))


# ** Featureset Exploration **
income_raw = data['income']
features_raw = data.drop('income', axis=1)
vs.distribution(data)

# Log-transform the skewed features
skewed = ['capital-gain', 'capital-loss']
features_log_transformed = pd.DataFrame(data=features_raw)
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))
vs.distribution(features_log_transformed, transformed=True)

scaler = MinMaxScaler()
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
features_log_minmax_transform = pd.DataFrame(data=features_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

# Show an example of a record with scaling applied
display(features_log_minmax_transform.head(n=5))

# Data Preprocessing
# One-hot encode
features_final = pd.get_dummies(features_log_minmax_transform)

# Encode the 'income_raw' data to numerical values
income = [1 if v.upper() == ">50K" else 0 for v in income_raw]

encoded = list(features_final.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_final, 
                                                    income, 
                                                    test_size=0.2,
                                                    random_state=0)

print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))


# Metrics and the Naive Predictor
accuracy = n_greater_50k / (n_greater_50k + n_at_most_50k)
recall = n_greater_50k / n_greater_50k
precision = n_greater_50k / (n_greater_50k + n_at_most_50k)

# F-score with beta = 0.5
fscore = (1 + 0.5**2) * (precision * recall / ((0.5**2 * precision) + recall))

print("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))


# Creating a Training and Predicting Pipeline
def train_predict(learner, sample_size, X_train, y_train, X_test, y_test):
    results_ = {}
    
    # Fit the learner
    start = time()
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time()
    
    # training time
    results_['train_time'] = end - start
        
    # Get the predictions on the test set then get predictions on the first 300 training samples
    start = time()
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time()
    
    # prediction time, accuracy and F-score
    results_['pred_time'] = end - start
    results_['acc_train'] = accuracy_score(y_train[:300], predictions_train)
    results_['acc_test'] = accuracy_score(y_test, predictions_test)
    results_['f_train'] = fbeta_score(y_train[:300], predictions_train, 0.5)
    results_['f_test'] = fbeta_score(y_test, predictions_test, 0.5)
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))

    return results_


# Model Evaluation
# Initialize the three models
clf_A = RandomForestClassifier(random_state=42)
clf_B = AdaBoostClassifier(random_state=42)
clf_C = GradientBoostingClassifier(random_state=42)

# number of samples for 1%, 10%, and 100% of the training data
samples_100 = len(y_train)
samples_10 = int(samples_100 / 10)
samples_1 = int(samples_10 / 10)

# Collect results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = train_predict(clf, samples, X_train, y_train, X_test, y_test)

# Run metrics visualization for the three supervised learning models chosen
vs.evaluate(results, accuracy, fscore)

print("Finding best AdaBoost parameters, this can take few minutes...")
# Improving Results
clf = AdaBoostClassifier(random_state=42)
parameters = {"n_estimators": [100, 250, 500], "learning_rate": [0.2, 0.5, 1.0]}
scorer = make_scorer(fbeta_score, beta=0.5)

# Perform grid search
grid_obj = GridSearchCV(clf, parameters, scoring=scorer, n_jobs=4)

# Get the best estimator
grid_fit = grid_obj.fit(X_train, y_train)
best_clf = grid_fit.best_estimator_

# Make predictions
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

# Report the best classifier parameters and before-and-afterscores
print(best_clf, "\n")
print("Unoptimized model\n------")
print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta=0.5)))
print("\nOptimized Model\n------")
print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("Final F-score on the testing data: {:.4f}\n".format(fbeta_score(y_test, best_predictions, beta=0.5)))


# Final Model Evaluation
# Extract the feature importances
model = best_clf.fit(X_train, y_train)
importances = model.feature_importances_
vs.feature_plot(importances, X_train, y_train)

# Reduce the feature space
X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:5]]]

# Make new predictions on the best model with less features
clf = (clone(best_clf)).fit(X_train_reduced, y_train)
reduced_predictions = clf.predict(X_test_reduced)

print("Final Model trained on full data\n------")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta=0.5)))
print("\nFinal Model trained on reduced data\n------")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, reduced_predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, reduced_predictions, beta=0.5)))
