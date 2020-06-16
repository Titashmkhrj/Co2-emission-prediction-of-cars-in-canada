import pandas as pd
from sklearn.preprocessing import (StandardScaler,
                                    Normalizer,
                                    MinMaxScaler,
                                    MaxAbsScaler,
                                    RobustScaler,
                                    PowerTransformer,
                                    QuantileTransformer)

class S_N_R() :
    '''
    This is a class to accomplish FEATURE-SCALLING, utilising the 'sklearn.preprocessing'.

    Availaible methods for this class are as follows :
    [[[FORMAT >> '<methd_name_of_this_class>'' using '<method_name_from_sklearn>'' : '<usage>'']]]
        * 'Standardization' using 'sklearn.preprocessing.StandardScaler' : Standardize features by removing the mean and scaling to unit variance.
        * 'Normalization' using 'sklearn.preprocessing.Normalizer' : Normalize samples individually to unit norm.
        * 'Min_Max_norm' using 'sklearn.preprocessing.MinMaxScaler' : Transform features by scaling each feature to a given range.
        * 'Max_abs_scaler' using 'sklearn.preprocessing.MaxAbsScaler' : Scale each feature by its maximum absolute value.
        * 'Robust_scaler' using 'sklearn.preprocessing.RobustScaler' : Scale features using statistics that are robust to outliers.
        * 'Power_transformer' using 'sklearn.preprocessing.PowerTransformer' : Apply a power transform featurewise to make data more Gaussian-like.
        * 'Quantile_transformer' using 'sklearn.preprocessing.QuantileTransformer' : Transform features using quantiles information.

    Class parameters are as follows :
        * x_train_data - data relating to the independent features of the training set,corresponding to the numerical continuous features.
        * x_test_data - data relating to the independent features of the testing set,corresponding to the numerical continuous features.
    '''

    def __init__ (self, x_train_data, x_test_data) :
        print(S_N_R.__doc__)
        self.x_train_data = x_train_data
        self.x_test_data = x_test_data
        self.scalling_operation = str(input('State the scalling operation :'))



    def _Standardization_ (self) : 
        # printing the docstring of StandardScaler, for informed user input.
        print(StandardScaler.__doc__)

        # making the return variables 'global', for further use of these in a project
        global scaled_x_train, scaled_x_test

        scaler = StandardScaler().fit(self.x_train_data)
        scaled_x_train = pd.DataFrame(scaler.transform(self.x_train_data))
        scaled_x_train.columns = list(self.x_train_data.columns)
        scaled_x_test = pd.DataFrame(scaler.transform(self.x_test_data))
        scaled_x_test.columns = list(self.x_test_data.columns)

        return scaled_x_train, scaled_x_test

    def _Normalization_ (self) : 
        # printing the docstring of Normalizer, for informed user input.
        print(Normalizer.__doc__)

        # making the return variables 'global', for further use of these in a project
        global scaled_x_train, scaled_x_test

        scaler = Normalizer().fit(self.x_train_data)
        scaled_x_train = pd.DataFrame(scaler.transform(self.x_train_data))
        scaled_x_train.columns = list(self.x_train_data.columns)
        scaled_x_test = pd.DataFrame(scaler.transform(self.x_test_data))
        scaled_x_test.columns = list(self.x_test_data.columns)

        return scaled_x_train, scaled_x_test

    def _Min_Max_norm_(self) :
        # printing the docstring of MinMaxScaler, for informed user input.
        print(MinMaxScaler.__doc__)

        # making the return variables 'global', for further use of these in a project
        global scaled_x_train, scaled_x_test

        scaler = MinMaxScaler().fit(self.x_train_data)
        scaled_x_train = pd.DataFrame(scaler.transform(self.x_train_data))
        scaled_x_train.columns = list(self.x_train_data.columns)
        scaled_x_test = pd.DataFrame(scaler.transform(self.x_test_data))
        scaled_x_test.columns = list(self.x_test_data.columns)

        return scaled_x_train, scaled_x_test

    def _Max_abs_scaler_(self) : 
        # printing the docstring of MaxAbsScaler, for informed user input.
        print(MaxAbsScaler.__doc__)

        # making the return variables 'global', for further use of these in a project
        global scaled_x_train, scaled_x_test

        scaler = MaxAbsScaler().fit(self.x_train_data)
        scaled_x_train = pd.DataFrame(scaler.transform(self.x_train_data))
        scaled_x_train.columns = list(self.x_train_data.columns)
        scaled_x_test = pd.DataFrame(scaler.transform(self.x_test_data))
        scaled_x_test.columns = list(self.x_test_data.columns)

        return scaled_x_train, scaled_x_test

    def _Robust_scaler_(self) :
        # printing the docstring of RobustScaler, for informed user input.
        print(RobustScaler.__doc__)

        # making the return variables 'global', for further use of these in a project
        global scaled_x_train, scaled_x_test

        scaler = RobustScaler(quantile_range=(25, 75)).fit(self.x_train_data)
        scaled_x_train = pd.DataFrame(scaler.transform(self.x_train_data))
        scaled_x_train.columns = list(self.x_train_data.columns)
        scaled_x_test = pd.DataFrame(scaler.transform(self.x_test_data))
        scaled_x_test.columns = list(self.x_test_data.columns)

        return scaled_x_train, scaled_x_test

    def _Power_transformer_(self) : 
        # printing the docstring of PowerTransformer, for informed user input.
        print(PowerTransformer.__doc__)
        # user inputs
        method = input('State the method to be used from ‘yeo-johnson’(works with positive and negative values) or ‘box-cox’(only works with strictly positive values) :')
        standardize = input('State Y/N if you want to apply zero-mean, unit-variance normalization to the transformed output :')

        # making the return variables 'global', for further use of these in a project
        global scaled_x_train, scaled_x_test

        if method == 'yeo-johnson' and  standardize == 'Y' :
            scaler = PowerTransformer(method='yeo-johnson', standardize=True).fit(self.x_train_data)
            scaled_x_train = pd.DataFrame(scaler.transform(self.x_train_data))
            scaled_x_train.columns = list(self.x_train_data.columns)
            scaled_x_test = pd.DataFrame(scaler.transform(self.x_test_data))
            scaled_x_test.columns = list(self.x_test_data.columns)
            return scaled_x_train, scaled_x_test
        
        elif method == 'yeo-johnson' and  standardize == 'N' :
            scaler = PowerTransformer(method='yeo-johnson', standardize=False).fit(self.x_train_data)
            scaled_x_train = pd.DataFrame(scaler.transform(self.x_train_data))
            scaled_x_train.columns = list(self.x_train_data.columns)
            scaled_x_test = pd.DataFrame(scaler.transform(self.x_test_data))
            scaled_x_test.columns = list(self.x_test_data.columns)
            return scaled_x_train, scaled_x_test

        elif method == 'box-cox' and  standardize == 'Y' :
            scaler = PowerTransformer(method='box-cox', standardize=True).fit(self.x_train_data)
            scaled_x_train = pd.DataFrame(scaler.transform(self.x_train_data))
            scaled_x_train.columns = list(self.x_train_data.columns)
            scaled_x_test = pd.DataFrame(scaler.transform(self.x_test_data))
            scaled_x_test.columns = list(self.x_test_data.columns)
            return scaled_x_train, scaled_x_test
        elif method == 'box-cox' and  standardize == 'N' :
            scaler = PowerTransformer(method='box-cox', standardize=False).fit(self.x_train_data)
            scaled_x_train = pd.DataFrame(scaler.transform(self.x_train_data))
            scaled_x_train.columns = list(self.x_train_data.columns)
            scaled_x_test = pd.DataFrame(scaler.transform(self.x_test_data))
            scaled_x_test.columns = list(self.x_test_data.columns)
            return scaled_x_train, scaled_x_test


    def _Quantile_transformer_(self) :
        # printing the docstring of QuantileTransformer, for informed user input.
        print(QuantileTransformer.__doc__)
        # user inputs
        rand_state = int(input('State the random state seed to be used :'))
        no_quantiles = int(input('State the number of quantiles to be computed :'))
        out_dist = str(input('State the marginal distribution for the transformed data. The choices are ‘uniform’ or ‘normal’ :'))
        ignr_implicit_zeros = bool(input("Only applies to sparse matrices, if 'True' the sparse entries of the matrix are discarded to compute the quantile statistics; if 'False', these entries are treated as zeros :"))

        # making the return variables 'global', for further use of these in a project
        global scaled_x_train, scaled_x_test

        if out_dist == 'normal' and ignr_implicit_zeros == True :
            scaler = QuantileTransformer(n_quantiles=no_quantiles, 
                                        output_distribution=out_dist, 
                                        ignore_implicit_zeros=ignr_implicit_zeros, 
                                        random_state=rand_state).fit(self.x_train_data)
            scaled_x_train = pd.DataFrame(scaler.transform(self.x_train_data))
            scaled_x_train.columns = list(self.x_train_data.columns)
            scaled_x_test = pd.DataFrame(scaler.transform(self.x_test_data))
            scaled_x_test.columns = list(self.x_test_data.columns)
            return scaled_x_train, scaled_x_test

        elif out_dist == 'normal' and ignr_implicit_zeros == False :
            scaler = QuantileTransformer(n_quantiles=no_quantiles, 
                                        output_distribution=out_dist, 
                                        ignore_implicit_zeros=ignr_implicit_zeros, 
                                        random_state=rand_state).fit(self.x_train_data)
            scaled_x_train = pd.DataFrame(scaler.transform(self.x_train_data))
            scaled_x_train.columns = list(self.x_train_data.columns)
            scaled_x_test = pd.DataFrame(scaler.transform(self.x_test_data))
            scaled_x_test.columns = list(self.x_test_data.columns)
            return scaled_x_train, scaled_x_test

        elif out_dist == 'uniform' and ignr_implicit_zeros == True :
            scaler = QuantileTransformer(n_quantiles=no_quantiles, 
                                        output_distribution=out_dist, 
                                        ignore_implicit_zeros=ignr_implicit_zeros, 
                                        random_state=rand_state).fit(self.x_train_data)
            scaled_x_train = pd.DataFrame(scaler.transform(self.x_train_data))
            scaled_x_train.columns = list(self.x_train_data.columns)
            scaled_x_test = pd.DataFrame(scaler.transform(self.x_test_data))
            scaled_x_test.columns = list(self.x_test_data.columns)
            return scaled_x_train, scaled_x_test
        else:
            scaler = QuantileTransformer(n_quantiles=no_quantiles, 
                                        output_distribution=out_dist, 
                                        ignore_implicit_zeros=ignr_implicit_zeros, 
                                        random_state=rand_state).fit(self.x_train_data)
            scaled_x_train = pd.DataFrame(scaler.transform(self.x_train_data))
            scaled_x_train.columns = list(self.x_train_data.columns)
            scaled_x_test = pd.DataFrame(scaler.transform(self.x_test_data))
            scaled_x_test.columns = list(self.x_test_data.columns)
            return scaled_x_train, scaled_x_test


    def operation(self) :

        if self.scalling_operation == 'Standardization' :
            return self._Standardization_()
        elif self.scalling_operation == 'Normalization' :
            return self._Normalization_()
        elif self.scalling_operation == 'MinMaxNormalization' :
            return self._Min_Max_norm_()
        elif self.scalling_operation == 'MaxAbsScalling' :
            return self._Max_abs_scaler_()
        elif self.scalling_operation == 'RobustScalling' :
            return self._Robust_scaler_()
        elif self.scalling_operation == 'PowerTransformation' :
            return self._Power_transformer_()
        elif self.scalling_operation == 'QuantileTransformation' :
            return self._Quantile_transformer_()
        else :
            raise Exception('Type of scalling operation not in scope, (refer "S_N_R.__doc__" ) .')

