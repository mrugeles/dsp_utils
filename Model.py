from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier 
from sklearn.linear_model import RidgeClassifierCV 
from sklearn.linear_model import SGDClassifier 
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import SGDRegressor 
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor 
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb

import pandas as pd
from time import time

class TrainInfo():
    def __init__(self, learner, train_time, pred_time, test_score, train_score):
        self.learner = None
        self.train_time = None
        self.pred_time = None
        self.test_score = None
        self.train_score = None


class Model():

    def init_regressors(self, seed):
        return {
            'AdaBoostRegressor': AdaBoostRegressor(random_state = seed),
            'BaggingRegressor': BaggingRegressor(random_state = seed),
            'ExtraTreesRegressor': ExtraTreesRegressor(random_state = seed),
            'GradientBoostingRegressor': GradientBoostingRegressor(random_state = seed),
            'RandomForestRegressor': RandomForestRegressor(random_state = seed),
            'XGBRegressor': xgb.XGBRegressor(),
            'LogisticRegression': LogisticRegression(random_state = seed),
            'PassiveAggressiveRegressor': PassiveAggressiveRegressor(random_state = seed),
            'SGDRegressor': SGDRegressor(random_state = seed),
            'KNeighborsRegressor': KNeighborsRegressor(),
            'MLPRegressor': MLPRegressor(random_state = seed),
            'DecisionTreeRegressor': DecisionTreeRegressor(random_state = seed),
            'ExtraTreeRegressor': ExtraTreeRegressor(random_state = seed),
        }   


    def init_classifiers(self, seed):
        return {
            'AdaBoostClassifier': AdaBoostClassifier(random_state = seed),
            'BaggingClassifier': BaggingClassifier(random_state = seed),
            'ExtraTreesClassifier': ExtraTreesClassifier(random_state = seed),
            'GradientBoostingClassifier': GradientBoostingClassifier(random_state = seed),
            'RandomForestClassifier': RandomForestClassifier(random_state = seed),
            'HistGradientBoostingClassifier': HistGradientBoostingClassifier(random_state = seed),
            'XGBClassifier': xgb.XGBClassifier(),
            'LogisticRegression': LogisticRegression(random_state = seed),
            'PassiveAggressiveClassifier': PassiveAggressiveClassifier(random_state = seed),
            'RidgeClassifier': RidgeClassifier(random_state = seed),
            'RidgeClassifierCV': RidgeClassifierCV(),
            'SGDClassifier': SGDClassifier(random_state = seed),
            'KNeighborsClassifier': KNeighborsClassifier(),
            'MLPClassifier': MLPClassifier(random_state = seed),
            'DecisionTreeClassifier': DecisionTreeClassifier(random_state = seed),
            'ExtraTreeClassifier': ExtraTreeClassifier(random_state = seed),
        }    

    ###
    #      This method trains a model.
    #
    #      Args:
    #       learner (classifier / regressor): Model to train.
    #       scorer (function): score function.
    #       X_train: (dataset): Features dataset to train the model.
    #       y_train: (dataset): Targe feature dataset to train the model.
    #       X_test: (dataset): Features dataset to test the model.
    #       y_test: (dataset): Targe feature dataset to test the model.
    #      Returns:
    #       learner (classifier / regressor): trained model.
    #       dfResults (dataset): Dataset with information about the trained model.
    ###

    def train_eval(self, learner, scorer, X_train, y_train, X_test, y_test):
        start = time()
        learner = learner.fit(X_train, y_train)
        end = time()

        train_time = end - start

        start = time()
        predictions_test = learner.predict(X_test)
        predictions_train = learner.predict(X_train)
        end = time() # Get end time

        pred_time = end - start

        train_score = scorer(y_train, predictions_train)

        test_score =  scorer(y_test, predictions_test)
    
        return TrainInfo(learner, train_time, pred_time, test_score, train_score)

    def split_data(self, df, target, config):
        from sklearn.model_selection import train_test_split
        features = df.drop([target], axis = 1)
        labels = df[target]
        return train_test_split(features, labels, test_size = config['test_size'], random_state = config['seed'], stratify=labels)

    def train_models(self, learners, scorer, X_train, X_test, y_train, y_test):
        trained_models = []

        for learner in list(learners.values()):
            train_info = self.train_eval(learner, scorer, X_train, y_train, X_test, y_test)
            trained_models += [train_info]            

        return trained_models

    ###
    #      This method use grid search to tune a learner.
    #
    #      Args:
    #       learner (classifier / regressor): learner to tune.
    #       parameters (dict): learner parameters.
    #       X_train: (dataset): Features dataset to train the model.
    #       y_train: (dataset): Targe feature dataset to train the model.
    #       X_test: (dataset): Features dataset to test the model.
    #       y_test: (dataset): Targe feature dataset to test the model.
    #      Returns:
    #       best_learner (classifier / regressor)): Classifier with the best score.
    #       default_score (float): Classifier score before being tuned.
    #       tuned_score (float): Classifier score after being tuned.
    #       cnf_matrix (float): Confusion matrix.
    ###
    def tune_learner(self, learner, parameters, scorer, X_train, X_test, y_train, y_test):

      c, r = y_train.shape
      labels = y_train.values.reshape(c,)

      grid_obj = GridSearchCV(learner, param_grid=parameters,  scoring=scorer, iid=False)
      grid_fit = grid_obj.fit(X_train, labels)
      best_learner = grid_fit.best_estimator_
      predictions = (learner.fit(X_train, labels)).predict(X_test)
      best_predictions = best_learner.predict(X_test)

      default_score = scorer(y_test, predictions)
      tuned_score = scorer(y_test, best_predictions)

      return best_learner, default_score, tuned_score
