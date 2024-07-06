#%% imports
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform, poisson

# load in the data
data = load_iris()

# isolate out the data we need
X = data.data
y = data.target
class_names = data.target_names
feature_names = data.feature_names

#%% perform a train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#%% fit a decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train,y_train)

#%% visualise the decision tree
fig = plt.figure(figsize=(16,8))
_ = plot_tree(clf,
              feature_names=feature_names,
              filled=True,
              class_names=class_names,
              fontsize=10)

#%% produce classification report
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

#%% - pre-pruning
# setup parameter space
parameters = {'max_depth':poisson(mu=2,loc=2),
              'max_leaf_nodes':poisson(mu=5,loc=5),
              'min_samples_split':uniform(),
              'min_samples_leaf':uniform()}

# create an instance of the randomized search object
rsearch = RandomizedSearchCV(DecisionTreeClassifier(random_state=42),
                             parameters, cv=10, n_iter=100, random_state=42)

# conduct randomised search over the parameter space
rsearch.fit(X_train,y_train)

#%% show best parameter configuration found for classifier
cls_params = rsearch.best_params_
cls_params['min_samples_split'] = np.ceil(cls_params['min_samples_split']*X_train.shape[0])
cls_params['min_samples_leaf'] = np.ceil(cls_params['min_samples_leaf']*X_train.shape[0])
cls_params

# extract best classifier
clf = rsearch.best_estimator_

#%% visualise the decision tree
fig = plt.figure(figsize=(16,8))
_ = plot_tree(clf,
              feature_names=feature_names,
              filled=True,
              class_names=class_names,
              fontsize=10)

#%% produce classification report
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

#%% - post-pruning
# step 1: fit a decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train,y_train)

# step 2: extract the set of cost complexity parameter alphas
ccp_alphas = clf.cost_complexity_pruning_path(X_train,y_train)['ccp_alphas']

# view the complete list of effective alphas
ccp_alphas.tolist()

#%% setup parameter space
parameters = {'ccp_alpha':ccp_alphas.tolist()}

# create an instance of the grid search object
gsearch = GridSearchCV(DecisionTreeClassifier(random_state=42), parameters, cv=10)

# step 3: conduct grid search over the parameter space
gsearch.fit(X_train,y_train)

#%% show best parameter configuration found for classifier
gsearch.best_params_

#%% extract best classifier
clf = gsearch.best_estimator_

# visualise the decision tree
fig = plt.figure(figsize=(16,8))
_ = plot_tree(clf,
              feature_names=feature_names,
              filled=True,
              class_names=class_names,
              fontsize=10)

#%% produce classification report
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))