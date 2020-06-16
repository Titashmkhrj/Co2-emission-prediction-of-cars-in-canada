import pandas as pandas
from sklearn.preprocessing import KBinsDiscretizer

class Cont_feat_handler :

    def __init__ (self, dataframe, feature) :
        self.dataframe = dataframe
        self.feature = feature
        self.feature_type = self.dataframe.dtypes[self.feature]


    def feature_binarizer (self) :
        # printing the docstring of KBinsDiscretizer, for informed user input.
        print(KBinsDiscretizer.__doc__)

        if self.feature_type != float and self.feature_type != int :
        	raise Exception('Wrong type of feature to use, apply a numerical continous feature')
        else :
        	no_of_bins = int(input('State the number of bins to be produced :'))
        	encoding = str(input("State the method used to encode the transformed result, from 'onehot', 'onehot-dense', 'ordinal' [(default='onehot'] :"))
        	strategy_to_be_applied = str(input("State the strategy used to define the widths of the bins, from 'uniform', 'quantile', 'kmeans'}, [default='quantile'] :"))

        	binarizer = KBinsDiscretizer(n_bins = no_of_bins, encode = encoding, strategy = strategy_to_be_applied)
        	return binarizer.fit_transform(self.dataframe[[self.feature]])