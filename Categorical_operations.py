import pandas as pd
import numpy as np
from sklearn.preprocessing import (Binarizer, 
									LabelBinarizer, 
									MultiLabelBinarizer, 
									LabelEncoder,
									OrdinalEncoder, 
									OneHotEncoder)

class Categorical_encoder:
    '''
    This is a class for encoding CATEGORICAL FEATURES(s), utilising the 'sklearn.preprocessing'.

    ------------ METHODS ------------- :
    [[[  FORMAT >> '<methd_name_of_this_class>'' using '<method_name_from_sklearn>'' : '<usage>'  ]]]

    * '_binarizer' using 'sklearn.preprocessing.Binarizer' : Binarize feature values to 0 or 1 according to a threshold.

    * '_label_binarizer' using 'sklearn.preprocessing.LabelBinarizer' : Binarize labels in a one-vs-all fashion.

    * '_multi_label_binarizer'using 'sklearn.preprocessing.MultiLabelBinarizer' : Transform between iterable of iterables and a multilabel       format.

    * '_label_encoder' using 'sklearn.preprocessing.LabelEncoder' : Encode target labels with value between 0 and n_classes-1.

    * '_one_hot_encoder' using 'sklearn.preprocessing.OneHotEncoder' : Encode categorical features as a one-hot numeric array.

    --------------- Class parameters ---------------- :

    * df - name of the dataframe of the data in use.
    feature  - name of the categorical feature in the dataframe, to  be encoded.
    enc_type - type of encoding to to be used for the feature, ranging from 'binary', 'label_binary', 'ml_binary', 'label_encoder', 'ohe'.
    '''

    def __init__(self, df, feature,):
        print(Categorical_encoder.__doc__)
        self.df = df
        self.feature = feature
        self.enc_type = str(input("State the type of encoding 'binary', 'label_binary', 'ml_binary', 'label_encoder', 'ohe' :"))
    
    def _binarizer(self):
        # printing the docstring of Binarizer, for informed user input.
        print(Binarizer.__doc__)
        # user inputs
        threshold_ = float(input('State the feature value below or equal to which are replaced by 0, above it by 1 :'))
        copy_ = bool(input('Set to False (by entering nothing) to perform inplace binarization and avoid a copy, default True :'))
        encoder = Binarizer(threshold = threshold_, copy = copy_).fit(self.df[[self.feature]])
        return encoder.transform(self.df[self.feature])
    
    def _label_binarizer(self):
        # printing the docstring of LabelBinarizer, for informed user input.
        print(LabelBinarizer.__doc__)
        # user inputs
        neg_label_ = int(input('State the value with which negative labels must be encoded :'))
        pos_label_ = int(input('State the value with which positive labels must be encoded :'))
        sparse_out = bool(input('State True if the returned array from transform is desired to be in sparse CSR format, otherwise enter nothing :'))
        encoder = LabelBinarizer(neg_label = neg_label_,
                                 pos_label= pos_label_,
                                 sparse_output= sparse_out).fit(self.df[self.feature])
        return encoder.transform(self.df[[self.feature]])

    def _multi_label_binarizer(self) :
        # printing the docstring of LabelEncoder, for informed user input.
        print(MultiLabelBinarizer.__doc__)
        encoder = MultiLabelBinarizer().fit(self.df[self.feature])
        encoded_feat_df = pd.DataFrame(encoder.transform(self.df[self.feature]), columns=encoder.classes_)
        return encoded_feat_df
    
    def _label_encoder(self):
        # printing the docstring of LabelEncoder, for informed user input.
        print(LabelEncoder.__doc__)
        encoder = LabelEncoder().fit(self.df[[self.feature]])
        return encoder.transform(self.df[[self.feature]])
        
#     def _ordinal_encoder(self):
#         # printing the docstring of OrdinalEncoder, for informed user input.
#         print(OrdinalEncoder.__doc__)
#         encoder = OrdinalEncoder().fit(self.df[[self.feature]])
#         return encoder.transform(self.df[[self.feature]])

    def _one_hot_encoder(self):
        # printing the docstring of OneHotEncoder, for informed user input.
        # print(OneHotEncoder.__doc__)
        # user inputs
        #handle_nan = str(input('Whether to raise an "error" or "ignore" if an unknown categorical feature is present during transform ( default="error") :'))
        #out_dtype = input('State the desired dtype of output :')
        # drop_ = str(input('State "first", "if_binary", or a array-like of shape (n_features,), default=None :'))
        
        encoder = OneHotEncoder(drop = None,
                            dtype = np.int,
                            handle_unknown = 'ignore',
                           sparse = False).fit(self.df[[self.feature]])
        
        return pd.DataFrame(encoder.transform(self.df[[self.feature]])), encoder.get_feature_names([self.feature])



    def operate(self):
        if self.enc_type == 'binary':
            return self._binarizer()
        elif self.enc_type == "label_binary":
            return self._label_binarizer()
        elif self.enc_type == "ml_binary":
            return self._multi_label_binarizer()
        elif self.enc_type == "label_encoder":
            return self._label_encoder()
        elif self.enc_type == "ohe":
            return self._one_hot_encoder()
        else:
            raise Exception("Encoding type not understood")