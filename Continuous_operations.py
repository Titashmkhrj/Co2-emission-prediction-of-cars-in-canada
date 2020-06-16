import pandas as pandas
from sklearn.preprocessing import (KBinsDiscretizer, Binarizer)

class Continuous_encoder :
    '''
    This is a class for encoding CATEGORICAL FEATURES(s), utilising the 'sklearn.preprocessing'.

    ------------ METHODS ------------- :
    [[[  FORMAT >> '<methd_name_of_this_class>'' using '<method_name_from_sklearn>'' : '<usage>'  ]]]

    * 'feature_discretizer' using 'sklearn.preprocessing.KBinsDiscretizer' : Binarize data (set feature values to 0 or 1) according to a threshold.

    * 'feature_binarizer' using 'sklearn.preprocessing.Binarizer' : Bin continuous data into intervals.

    Class parameters are as follows :

    * df - name of the dataframe of the data in use.
    * feature  - name of the categorical feature in the dataframe, to  be encoded.
    * feature_type - data type of the feature
    '''

    def __init__ (self, dataframe, feature) :
        self.dataframe = dataframe
        self.feature = feature
        self.feature_type = dtype(self.dataframe[self.feature])


    def feature_discretizer (self) :
        # printing the docstring of KBinsDiscretizer, for informed user input.
        print(KBinsDiscretizer.__doc__)

        if self.feature_type != float and self.feature_type != int :
        	raise Exception('Wrong type of feature to use, apply a numerical continous feature')
        else :
        	no_of_bins = int(input('State the number of bins to be produced :'))
        	encoding = str(input("State the method used to encode the transformed result, from 'onehot', 'onehot-dense', 'ordinal' [(default='onehot'] :"))
        	strategy_to_be_applied = str(input("State the strategy used to define the widths of the bins, from 'uniform', 'quantile', 'kmeans'}, [default='quantile'] :"))

        	binarizer = KBinsDiscretizer(n_bins = no_of_bins, encode = encoding, strategy = strategy_to_be_applied)
        	return binarizer.fit_transform(self.dataframe[self.feature])

    def feature_binarizer (self) :
        # printing the docstring of binarizer, for informed user inputs.
        print(Binarizer.__doc__)
        
        # user inputs
        threshold_ = float(input('State the feature value below or equal to which are replaced by 0, above it by 1 :'))
        copy_ = bool(input('Set to False (by entering nothing) to perform inplace binarization and avoid a copy, default True :'))

        encoder = Binarizer(threshold = threshold_, copy = copy_).fit(self.df[[self.feature]])
        return encoder.transform(self.df[[self.feature]])