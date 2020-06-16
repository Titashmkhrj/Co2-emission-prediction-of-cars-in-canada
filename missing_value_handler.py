import pandas as pd
import numpy as np

 # explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer  # noqa
# now you can import normally from sklearn.impute
from sklearn.impute import (SimpleImputer,
							IterativeImputer,
							MissingIndicator,
							KNNImputer)
class Missing_value_handling :
	
	'''
	This is a class for managing MISSING-VALUE(s), utilising the 'sklearn.impute'.

	------------ METHODS ------------- :
	[[[  FORMAT >> '<methd_name_of_this_class>'' using '<method_name_from_sklearn>'' : '<usage>''  ]]]

	* 'Missing_boolean_indicator' using 'sklearn.impute.MissingIndicator' :Binary indicators for missing values.
	* 'Simple_imputer' using 'sklear.impute.SiimpleImputer' : Imputation transformer for completing missing values.

	* 'Iterative_imputer' using 'sklear.impute.IterativeImputer' : Multivariate imputer that estimates each feature from 
		all the others. A strategy for imputing missing values by modeling each feature with missing values as a 
		function of other features in a round-robin fashion.

	* 'KNN_imputer' using 'sklearn.impute.KNNImputer' :Imputation for completing missing values using k-Nearest Neighbors.
		Each sample's missing values are imputed using the mean value from n_neighbors nearest neighbors found in the
		training set. Two samples are close if the features that neither is missing are close. 

	Class parameters are as follows :
	* dataframe - name of the dataframe of the data in use.
	* feature - name of the feature in the dataframe, to detect and manage missing-values.
		
	* missing_value - the placeholder for missing-value in that feature, 
	e.g : Nan, np.nan, or any iniger, or some specific string.

	'''
	def __init__(self, dataframe, feature, missing_value = np.NaN):

		self.dataframe = dataframe
		self.feature = feature
		self.missing_value = missing_value
		self.imputation_strategy = str(input('State the imputation strategy to be used from "indication", "simple", "iterative", "knn" :'))


	
	def Missing_boolean_indicator(self) :
		# priinting the docstring of MissingIndicator, for informed user usage.
		print(MissingIndicator.__doc__)

		missing_value_boolean_indicator = MissingIndicator(missing_values = self.missing_value)
		return missing_value_boolean_indicator.fit_transform(self.dataframe[[self.feature]])

	
	def Simple_imputer(self) :
		# priinting the docstring of SimpleImputer, for informed user usage.
		print(SimpleImputer.__doc__)

		feature_type = str(input("Type of feature is categorical or continous ?"))
		strategy = str(input("strategy to be used is :"))
		
		if feature_type == 'continous' and strategy in ("mean", "median", "most_frequent") :
			imputer =  SimpleImputer(missing_values = self.missing_value, strategy = strategy)
			return imputer.fit_transform(self.dataframe[[self.feature]])

		elif feature_type == 'categorical' and strategy in ("constant", "most_frequent") :
			if strategy == "constant" :
				fill_value = input("Value to be used as a 'fill_value' is :")
				imputer =  SimpleImputer(missing_values = self.missing_value, strategy = strategy, fill_value = fill_value)
				return imputer.fit_transform(self.dataframe[[self.feature]])
			else:
				imputer =  SimpleImputer(missing_values = self.missing_value, strategy = strategy)
				return imputer.fit_transform(self.dataframe[[self.feature]])
		else :
			raise Exception("Invalid type of strategy or feature type for the use of SimpleImputer please refer to -  https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html#sklearn.impute.SimpleImputer")

	
	def Iterative_imputer(self) :
		# priinting the docstring of IterativeImputer, for informed user usage.
		print(IterativeImputer.__doc__)

		# user inputs 
		initial_strategy = str(input("Initial strategy to be used is :"))
		max_iterations = int(input("State the number of maximum iterations to be used :"))
		n_nearest_features = int(input("State the number of other features to use to estimate the missing values of each feature column."))
		imputation_order = str(input("State the order in which the features will be imputed. Possible values: 'ascending', 'descending', 'roman', 'arabic', 'random' :"))
		random_state_seed = int(input('State the random state of the estimator :'))
		skip_complete = bool(input("True / False ? :"))
		sample_posterior = bool(input("True / False ? :"))
		feature_space_to_fit = str(input("State the feature names for the fitting of the estimator, for example ['feature_1'], ['feature_2']:"))
		
		if initial_strategy in ("mean", "median", "most_frequent", "constant") :
			imputer = IterativeImputer(missing_values = self.missing_value,
										max_iter = max_iterations,
										random_state = random_state_seed,
										initial_strategy = initial_strategy,
										n_nearest_features = n_nearest_features,
										imputation_order = imputation_order,
										skip_complete = skip_complete,
										sample_posterior= sample_posterior		
									   )
			imputer.fit(self.dataframe[feature_space_to_fit])
			return imputer.transform(self.dataframe[[self.feature]])

		else:
			raise Exception('''Invalid type of strategy for initializing the IterativeImputer;
    		please refer to -  https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html#sklearn.impute.IterativeImputer''')

	
	def KNN_imputer(self) :
		# priinting the docstring of KNNImputer, for informed user usage.
		print(KNNImputer.__doc__)

		# user inputs
		neighbors = int(input('State the number of neighboring samples to use for imputation :'))
		weight_func = str(input('State the weight function to be applied in prediction :'))
		metrics = str(input('State the distance metric to apply for searching neighbors :'))
		imputer = KNNImputer(missing_values = self.missing_value, 
							n_neighbors= neighbors,
							weights = weight_func, 
							metric = metrics, 
							)
		return imputer.fit_transform(self.dataframe[[self.feature]])


    def operate(self):
        if self.imputation_strategy == 'indication':
            return self.Missing_boolean_indicator()
        elif self.imputation_strategy == "simple":
            return self.Simple_imputer()
        elif self.imputation_strategy == "iterative":
            return self.Iterative_imputer()
        elif self.imputation_strategy == "knn":
            return self.KNN_imputer()
        else:
            raise Exception("Imputation strategy not understood.")



