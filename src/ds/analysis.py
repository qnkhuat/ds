import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import pandas as pd
from time import time

# filter warnings
import warnings
warnings.filterwarnings('ignore')

#Common Model Helpers
from sklearn import feature_selection 
from sklearn import model_selection
from sklearn import metrics



def model_comparison(x,y,show=True):
    """ Copy from : https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy/notebook
	Compare with various machine learning model
    """
    from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
    from xgboost import XGBClassifier


    MLA = [
	#Ensemble Methods
	ensemble.AdaBoostClassifier(),
	ensemble.BaggingClassifier(),
	ensemble.ExtraTreesClassifier(),
	ensemble.GradientBoostingClassifier(),
	ensemble.RandomForestClassifier(),

	#Gaussian Processes
	gaussian_process.GaussianProcessClassifier(),

	#GLM
	linear_model.LogisticRegressionCV(),
	linear_model.PassiveAggressiveClassifier(),
	linear_model.RidgeClassifierCV(),
	linear_model.SGDClassifier(),
	linear_model.Perceptron(),

	#Navies Bayes
	naive_bayes.BernoulliNB(),
	naive_bayes.GaussianNB(),

	#Nearest Neighbor
	neighbors.KNeighborsClassifier(),

	#SVM
	svm.SVC(probability=True),
	svm.NuSVC(probability=True),
	svm.LinearSVC(),

	#Trees
	tree.DecisionTreeClassifier(),
	tree.ExtraTreeClassifier(),

	#Discriminant Analysis
	discriminant_analysis.LinearDiscriminantAnalysis(),
	discriminant_analysis.QuadraticDiscriminantAnalysis(),


	#xgboost: http://xgboost.readthedocs.io/en/latest/model.html
	XGBClassifier()
    ]


    #split dataset in cross-validation with this splitter class: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit
    #note: this is an alternative to train_test_split
    cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3,
            train_size = .7, random_state = 0 ) # run model

    #create table to compare MLA metrics
    MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean',
            'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']

    MLA_compare = pd.DataFrame(columns = MLA_columns)


    #create table to compare MLA predictions
    MLA_predict = y.copy()

    #index through MLA and save performance to table
    row_index = 0

    for alg in MLA:
	#set name and parameters
    	MLA_name = alg.__class__.__name__
    	MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    	MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())

    	#score model with cross validation: 
    	cv_results = model_selection.cross_validate(alg, x,
                    y, cv  = cv_split)

    	MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    	MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    	MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()
    	#if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
    	MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3   #let's know the worst that can happen!


    	#save MLA predictions - see section 6 for usage
    	alg.fit(x, y)
    	MLA_predict[MLA_name] = alg.predict(x)

    	row_index+=1

    MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)

    if show :
        plt.figure(figsize=(15,6))
        sns.barplot(x='MLA Test Accuracy Mean', y = 'MLA Name', data = 
                MLA_compare, color = 'm')
        plt.show()

    return MLA_compare

def select_feature(alg,x,y,cv_split=None,is_print=True):

    if cv_split == None:
        cv_split = model_selection.ShuffleSplit(n_splits = 10, 
                test_size = .3, train_size = .7, random_state = 0 )
    
    base_results = model_selection.cross_validate(alg,x,y,cv=cv_split)

    if is_print:
        print('BEFORE DT RFE Training Shape Old: ', x.shape) 
        print('BEFORE DT RFE Training Columns Old: ', x.columns.values)

        print("BEFORE DT RFE Training w/bin score mean: {:.2f}". format(
            base_results['train_score'].mean()*100)) 
        print("BEFORE DT RFE Test w/bin score mean: {:.2f}". format(
            base_results['test_score'].mean()*100))
        print("BEFORE DT RFE Test w/bin score 3*std: +/- {:.2f}". format(
            base_results['test_score'].std()*100*3))
        print('-'*10)


    #feature selection
    dtree_rfe = feature_selection.RFECV(alg, step = 1, scoring = 'accuracy', cv = cv_split)
    dtree_rfe.fit(x, y)

    #transform x&y to reduced features and fit new model
    #alternative: can use pipeline to reduce fit and transform steps: http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
    # the optimal columns
    X_rfe = x.columns.values[dtree_rfe.get_support()]
    rfe_results = model_selection.cross_validate(alg, x[X_rfe], y, cv  = cv_split)

    if is_print:
        print('AFTER DT RFE Training Shape New: ', x[X_rfe].shape) 
        print('AFTER DT RFE Training Columns New: ', X_rfe)

        print("AFTER DT RFE Training w/bin score mean: {:.2f}". format(
            rfe_results['train_score'].mean()*100)) 
        print("AFTER DT RFE Test w/bin score mean: {:.2f}". format(
            rfe_results['test_score'].mean()*100))
        print("AFTER DT RFE Test w/bin score 3*std: +/- {:.2f}". format(
            rfe_results['test_score'].std()*100*3))
        print('-'*10)


    return X_rfe


def param_search(alg,x,y,params=None,cv_split=None):

    #Hyperparameter Tune with GridSearchCV: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    grid_n_estimator = [10, 50, 100, 300]
    grid_ratio = [.1, .25, .5, .75, 1.0]
    grid_learn = [.01, .03, .05, .1, .25]
    grid_max_depth = [2, 4, 6, 8, 10, None]
    grid_min_samples = [5, 10, .03, .05, .10]
    grid_criterion = ['gini', 'entropy']
    grid_bool = [True, False]
    grid_seed = [0]

    alg_name = alg.__class__.__name__
    if cv_split is None:
        cv_split = model_selection.ShuffleSplit(n_splits = 10, 
                test_size = .3, train_size = .7, random_state = 0 )


    if alg_name == 'AdaBoostClassifier':
        params = [{
		#AdaBoostClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
		'n_estimators': grid_n_estimator, #default=50
		'learning_rate': grid_learn, #default=1
		#'algorithm': ['SAMME', 'SAMME.R'], #default=’SAMME.R
		'random_state': grid_seed
		}]

    elif alg_name == 'BaggingClassifier':
        params = [{
		#BaggingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier
		'n_estimators': grid_n_estimator, #default=10
		'max_samples': grid_ratio, #default=1.0
		'random_state': grid_seed
        }]

    elif alg_name == 'ExtraTreesClassifier':# ExtraTreesClassifier 
        params = [{
		#ExtraTreesClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier
		'n_estimators': grid_n_estimator, #default=10
		'criterion': grid_criterion, #default=”gini”
		'max_depth': grid_max_depth, #default=None
		'random_state': grid_seed
         }]

    elif alg_name == 'GradientBoostingClassifier':# GradientBoostingClassifier

        params = [{    
                #GradientBoostingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
		'max_iter_predict': grid_n_estimator, #default: 100
		'random_state': grid_seed
        }]

    elif alg_name == 'RandomForestClassifier':# RandomForestClassifier 
	#RandomForestClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
        params = [{
		'n_estimators': grid_n_estimator, #default=10
		'criterion': grid_criterion, #default=”gini”
		'max_depth': grid_max_depth, #default=None
		'oob_score': [True], #default=False -- 12/31/17 set to reduce runtime -- The best parameter for RandomForestClassifier is {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100, 'oob_score': True, 'random_state': 0} with a runtime of 146.35 seconds.
		'random_state': grid_seed
         }]

    elif alg_name == 'GaussianProcessClassifier':# GaussianProcessClassifier
        params = [{    
		'max_iter_predict': grid_n_estimator, #default: 100
		'random_state': grid_seed
        }]

    elif alg_name == 'LogisticRegressionCV':# LogisticRegressionCV 
	#LogisticRegressionCV - http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV
        params = [{
		'fit_intercept': grid_bool, #default: True
		#'penalty': ['l1','l2'],
		'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], #default: lbfgs
		'random_state': grid_seed
         }]

    elif alg_name == 'KNeighborsClassifier':# KNeighborsClassifier
        params =[{
		#KNeighborsClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
		'n_neighbors': [1,2,3,4,5,6,7], #default: 5
		'weights': ['uniform', 'distance'], #default = ‘uniform’
		'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }]
    elif alg_name == 'SVC':# SVC
        params = [{
		#SVC - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
		#http://blog.hackerearth.com/simple-tutorial-svm-parameter-tuning-python-r
		#'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
		'C': [1,2,3,4,5], #default=1.0
		'gamma': grid_ratio, #edfault: auto
		'decision_function_shape': ['ovo', 'ovr'], #default:ovr
		'probability': [True],
		'random_state': grid_see
         }]

    elif alg_name == 'XGBClassifier':# XGBClassifier
        params = [{
		#XGBClassifier - http://xgboost.readthedocs.io/en/latest/parameter.html
		'learning_rate': grid_learn, #default: .3
		'max_depth': [1,2,4,6,8,10], #default 2
		'n_estimators': grid_n_estimator, 
		'seed': grid_seed  
        }]   

    else:
        if params is None:
            print("Don't have param for this algorithm, input params search manually")
            return None
        else :
            pass


    print("Param search: ",params)

    start = time()        
    best_search = model_selection.GridSearchCV(estimator = alg, 
	    param_grid = params, cv = cv_split, scoring = 'roc_auc')

    best_search.fit(x,y)
    run = time() - start

    best_param = best_search.best_params_
    print('The best parameter for {} is {} with a runtime of {:.2f} seconds.'.format(
	    alg_name, best_param, run))

    print('-'*10)

    return best_param
	
