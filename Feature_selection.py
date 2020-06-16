import pandas as pandas

from sklearn.feature_selection import (GenericUnivariateSelect,
										SelectKBest, 
										SelectPercentile, 
										SelectFpr, 
										SelectFdr, 
										SelectFwe,
										SelectFromModel,
										RFE,
										RFECV,
										VarianceThreshold, 
										chi2, 
										f_classif, 
							     	    mutual_info_classif, 
									    f_regression,
								    	mutual_info_regression)
class Feature_selection() :
	'''
	This is a class to accomplish FEATURE-SELECTION, utilising 'sklearn.feature_selection'.

	Availaible methods for this class are as follows :
	[[[FORMAT >> '<methd_name_of_this_class>'' using '<method_name_from_sklearn>'' : '<usage>'']]]
		* 'Generic_Univariate_Selector' using 'sklearn.feature_selection.GenericUnivariateSelect' : Univariate feature selector with configurable strategy.
		* 'Select_Kbest' using 'sklearn.feature_selection.SelectKBest' : Select features according to the k highest scores.
		* 'Select_percentile using 'sklearn.feature_selection.SelectPercentile' : Select features according to a percentile of the highest scores.
		* 'Select_FPR' using 'sklearn.feature_selection.SelectFpr' : Filter: Select the pvalues below alpha based on a FPR test.
		* 'Select_FDR' using 'sklearn.feature_selection.SelectFdr' : Filter: Select the p-values for an estimated false discovery rate
		* 'Select_FWE' using 'sklearn.feature_selection.SelectFwe' : Filter: Select the p-values corresponding to Family-wise error rate
		* 'Select_from_model' using 'sklearn.feature_selection.SelectFromModel' : Meta-transformer for selecting features based on importance weights.
		* 'Varience_threshold' using 'sklearn.feature_selection.VarianceThreshold' : Feature selector that removes all low-variance features.
		* 'Recursive_Feat_Elimination' using 'sklearn.feature_selection.RFE' : Feature ranking with recursive feature elimination.
		* 'Recursive_Feat_Elimination_and_CV_feat_selection' using 'sklearn.feature_selection.RFECV' : Feature ranking with recursive feature elimination and cross-validated selection of the best number of features.

	Class parameters are as follows :
		* x_data : data relating to the independent features of our dataset.
		* y_data : data relating to the dependent features of our dataset.
	'''

	def __init__(self, x_data, y_data):
		self.x_data = x_data
		self.y_data = y_data
		
	def Generic_Univariate_Selector (self) :
		# printing the docstring of GenericUnivariateSelect, for informed user input.
		print(GenericUnivariateSelect.__doc__)
		
		# making the return variables 'global', for further use of these in a project
        global x_new

        # user inputs		
		prob_type = int(input('State the type of problem from Regression[0] / Classification[1] :'))
		score_function = str(input('State the callable function taking two arrays X and y; returning a pair of arrays - scores, pvalues (For modes ‘percentile’ or ‘kbest’ it can return a single array scores) :'))
		mode = str(input('State the feature selection mode from ‘percentile’, ‘k_best’, ‘fpr’, ‘fdr’, ‘fwe’ :'))
		parameter = int(input('State the parameter of the corresponding mode :'))
		
		if prob_type == 1 and score_function in ('chi2', 'f_classif', 'mutual_info_classif') :
			selector = GenericUnivariateSelect(score_func = score_function, mode = mode, param = parameter)
			x_new = selector.fit_transform(self.x_data, self.y_data)
			return x_new

		elif prob_type == 0 and score_function in ('f_regression', 'mutual_info_regression') :
			selector = GenericUnivariateSelect(score_func = score_function, mode = mode, param = parameter)
			x_new = selector.fit_transform(self.x_data, self.y_data)
			return x_new

		else :
			rise Exception('Wrong choice of score function for the type of problem; refer - https://scikit-learn.org/stable/modules/feature_selection.html#feature-selection')


	def Select_Kbest (self) :
		# printing the docstring of SelectKBest, for informed user input.
		print(SelectKBest.__doc__)
		
		# making the return variables 'global', for further use of these in a project
        global x_new

        # user inputs 
		prob_type = int(input('State the type of problem from Regression[0] / Classification[1] :'))
		score_function = str(input('State the callable function taking two arrays X and y; returning a pair of arrays - scores, pvalues (For modes ‘percentile’ or ‘kbest’ it can return a single array scores) :'))
		no_K = int(input("State the number of top features to select (The 'all' option bypasses selection, for use in a parameter search) :"))

		if prob_type == 1 and score_function in ('chi2', 'f_classif', 'mutual_info_classif') :
			selector = SelectKBest(score_func = score_function, K = no_K)
			x_new = selector.fit_transform(self.x_data, self.y_data)
			return x_new
		elif prob_type == 0 and score_function in ('f_regression', 'mutual_info_regression') :
			selector = SelectKBest(score_func = score_function, K = no_K)
			x_new = selector.fit_transform(self.x_data, self.y_data)
			return x_new
		else :
			rise Exception('Wrong choice of score function for the type of problem; refer - https://scikit-learn.org/stable/modules/feature_selection.html#feature-selection')

	def Select_percentile(self) :
		# printing the docstring of SelectPercentile, for informed user input.
		print(SelectPercentile.__doc__)
		
		# making the return variables 'global', for further use of these in a project
        global x_new

        # user inputs

		prob_type = int(input('State the type of problem from Regression[0] / Classification[1] :'))
		score_function = str(input('State the callable function taking two arrays X and y; returning a pair of arrays - scores, pvalues (For modes ‘percentile’ or ‘kbest’ it can return a single array scores) :'))
		percent_feat = int(input('State the percent of features to keep :'))

		if prob_type == 1 and score_function in ('chi2', 'f_classif', 'mutual_info_classif') :
			selector = SelectPercentile(score_func = score_function, percentile = percent_feat)
			x_new = selector.fit_transform(self.x_data, self.y_data)
			return x_new
		elif prob_type == 0 and score_function in ('f_regression', 'mutual_info_regression') :
			selector = SelectPercentile(score_func = score_function, percentile = percent_feat)
			x_new = selector.fit_transform(self.x_data, self.y_data)
			return x_new
		else :
			rise Exception('Wrong choice of score function for the type of problem; refer - https://scikit-learn.org/stable/modules/feature_selection.html#feature-selection')


	def Select_FPR(self) :
		# printing the docstring of SelectFpr, for informed user input.
		print(SelectFpr.__doc__)
		
		# making the return variables 'global', for further use of these in a project
        global x_new

        # user inputs
		prob_type = int(input('State the type of problem from Regression[0] / Classification[1] :'))
		score_function = str(input('State the callable function taking two arrays X and y; returning a pair of arrays - scores, pvalues (For modes ‘percentile’ or ‘kbest’ it can return a single array scores) :'))
		alpha_ = float(input('State the highest p-value for features to be kept :'))

		if prob_type == 1 and score_function in ('chi2', 'f_classif', 'mutual_info_classif') :
			selector = SelectFpr(score_func = score_function, alpha = aplha_)
			x_new = selector.fit_transform(self.x_data, self.y_data)
			return x_new
		elif prob_type == 0 and score_function in ('f_regression', 'mutual_info_regression') :
			selector = SelectFpr(score_func = score_function, alpha = aplha_)
			x_new = selector.fit_transform(self.x_data, self.y_data)
			return x_new
		else :
			rise Exception('Wrong choice of score function for the type of problem; refer - https://scikit-learn.org/stable/modules/feature_selection.html#feature-selection')


	def Select_FDR(self) :
		# printing the docstring of SelectFdr, for informed user input.
		print(SelectFdr.__doc__)
		
		# making the return variables 'global', for further use of these in a project
        global x_new

        # user inputs
		prob_type = int(input('State the type of problem from Regression[0] / Classification[1] :'))
		score_function = str(input('State the callable function taking two arrays X and y; returning a pair of arrays - scores, pvalues (For modes ‘percentile’ or ‘kbest’ it can return a single array scores) :'))
		alpha_ = float(input('State the highest uncorrected p-value for features to keep :'))

		if prob_type == 1 and score_function in ('chi2', 'f_classif', 'mutual_info_classif') :
			selector = SelectFdr(score_func = score_function, alpha = aplha_)
			x_new = selector.fit_transform(self.x_data, self.y_data)
			return x_new
		elif prob_type == 0 and score_function in ('f_regression', 'mutual_info_regression') :
			selector = SelectFdr(score_func = score_function, alpha = aplha_)
			x_new = selector.fit_transform(self.x_data, self.y_data)
			return x_new
		else :
			rise Exception('Wrong choice of score function for the type of problem; refer - https://scikit-learn.org/stable/modules/feature_selection.html#feature-selection')


	def Select_FWE(self) :
		# printing the docstring of SelectFwe, for informed user input.
		print(SelectFwe.__doc__)
		
		# making the return variables 'global', for further use of these in a project
        global x_new

        # user inputs
		prob_type = int(input('State the type of problem from Regression[0] / Classification[1] :'))
		score_function = str(input('State the callable function taking two arrays X and y; returning a pair of arrays - scores, pvalues (For modes ‘percentile’ or ‘kbest’ it can return a single array scores) :'))
		alpha_ = float(input('State the highest uncorrected p-value for features to keep :'))

		if prob_type == 1 and score_function in ('chi2', 'f_classif', 'mutual_info_classif') :
			selector = SelectFwe(score_func = score_function, alpha = aplha_)
			x_new = selector.fit_transform(self.x_data, self.y_data)
			return x_new
		elif prob_type == 0 and score_function in ('f_regression', 'mutual_info_regression') :
			selector = SelectFwe(score_func = score_function, alpha = aplha_)
			x_new = selector.fit_transform(self.x_data, self.y_data)
			return x_new
		else :
			rise Exception('Wrong choice of score function for the type of problem; refer - https://scikit-learn.org/stable/modules/feature_selection.html#feature-selection')


	def Select_from_model(self) :
		# printing the docstring of SelectFromModel, for informed user input.
		print(SelectFromModel.__doc__)
		
		# making the return variables 'global', for further use of these in a project
        global x_new

        # user inputs
		estimator_object = input('State the base estimator "object" from which the transformer is built. This can be both a fitted (if prefit is set to True) or a non-fitted estimator. The estimator must have either a feature_importances_ or coef_ attribute after fitting :')
		threshold = input('(OPTIONAL and by_default None)State the threshold value to use for feature selection. Features whose importance is greater or equal are kept while the others are discarded :')
		prefit = bool(input('State True/False whether a prefit model is expected to be passed into the constructor directly or not :'))
		norm_order = input('(ranging from -inf to inf, by_default 1)State the order of the norm used to filter the vectors of coefficients below threshold in the case where the coef_ attribute of the estimator is of dimension 2 :')
		max_features = int(input('(OPTIONAL)State the maximum number of features selected scoring above threshold :'))

		selector = SelectFromModel(estimator = estimator_object, 
									threshold = threshold, 
									prefit = prefit, 
									norm_order = norm_order, 
									max_features = max_features)
		x_new = selector.fit_transform(self.x_data, self.y_data)
		return x_new


	def Varience_threshold(self) :
		# printing the docstring of VarianceThreshold, for informed user input.
		print(VarianceThreshold.__doc__)
		
		# making the return variables 'global', for further use of these in a project
        global x_new

        # user inputs
		threshold = float(input('(OPTIONAL, float)State the threshold, where features with a training-set variance lower than this threshold will be removed. The default is to keep all features with non-zero variance, i.e. remove the features that have the same value in all samples :'))

		selector = VarianceThreshold(threshold = threshold)
		x_new = selector.fit_transform(self.x_data, self.y_data)
		return x_new


	def Recursive_Feat_Elimination(self) :
		# printing the docstring of RFE, for informed user input.
		print(RFE.__doc__)
		
		# making the return variables 'global', for further use of these in a project
        global x_new

        # user inputs
		estimator_object = input('State a supervised learning estimator object, with a fit method that provides information about feature importance either through a coef_ attribute or through a feature_importances_ attribute :')
		no_features = int(input('(int/ by_defaut None)State the number of features to select. If None, half of the features are selected :'))
		step = input('(OPTIONAL, int or float or by_default=1) State the step; if step >= 1, then step corresponds to the (integer) number of features to remove at each iteration. If 0.0 < step < 1.0, then step corresponds to the percentage (rounded down) of features to remove at each iteration :')

		selector = RFE(estimator = estimator_object, n_features_to_select = no_features, step = step)
		x_new = selector.fit_transform(self.x_data, self.y_data)
		return x_new

	def Recursive_Feat_Elimination_and_CV_feat_selection(self) :
		# printing the docstring of RFECV, for informed user input.
		print(RFECV.__doc__)
		
		# making the return variables 'global', for further use of these in a project
        global x_new

        # user inputs
		estimator_object = input('State a supervised learning estimator object, with a fit method that provides information about feature importance either through a coef_ attribute or through a feature_importances_ attribute :')
		step = input('(OPTIONAL, int or float or by_default=1) State the step; if step >= 1, then step corresponds to the (integer) number of features to remove at each iteration. If 0.0 < step < 1.0, then step corresponds to the percentage (rounded down) of features to remove at each iteration :')
		min_no_of_feat = int(input('State the minimum number of features to be selected :'))
		cross_val = input('State the the CV splitting strategy. Possible inputs are None(to use the default 5-fold CV), int(to specify the no of folds), a CV splitter object, or an iterable yielding (train, test) splits as arrays of indices :')
		scoring = iput('State a string (see model evaluation documentation) or a scorer callable object / function with signature scorer(estimator, X, y) :')
		no_jobs = input('(OPTIONAL, int or None, by_default=None) State the number of cores to run in parallel while fitting across folds :')

		selector = RFECV(estimator = estimator_object, 
						step = step, 
						min_features_to_select = min_no_of_feat, 
						cv = cross_val, 
						scoring = scoring, 
						n_job = no_jobs)
		
		x_new = selector.fit_transform(self.x_data, self.y_data)
		return x_new