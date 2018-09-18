import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import pandas as pd

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

    	#score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
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

        #print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html
    MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)

    if show :
        plt.figure(figsize=(12,6))
        sns.barplot(x='MLA Test Accuracy Mean', y = 'MLA Name', data = MLA_compare, color = 'm')
        plt.show()

    return MLA_compare
