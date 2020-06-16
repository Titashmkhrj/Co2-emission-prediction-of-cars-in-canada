from sklearn.model_selection import (GridSearchCV,
                                     RandomizedSearchCV)


class Hyper_param_optimisation:
    '''
    some documentation of Hyper_param_optimisation class.
    '''
    def __init__(self,
                x_train_data,
                y_train_data,
                model_obj_list, 
                model_param_dict               
                ):
        print(Hyper_param_optimisation.__doc__)

        self.x_train_data = x_train_data
        self.y_train_data = y_train_data
        self.model_obj_list = model_obj_list
        self.model_param_dict = model_param_dict



    def  Grid_Search(self, CV = 5):
        # printing the docstring of GridSearchCV, for informed user input.
        print(GridSearchCV.__doc__)
        
        # making the return variables 'global', for further use of these in a project
        global Hyper_param_optimized_models
        Hyper_param_optimized_models = []

        # user inputs 
        # scoring = str(input('State the scoring parameter or a list of scoring parameters, (default=None) :'))
        #no_jobs = int(input('State the no of jobs to run in parallel, (default=None) :'))
        #refit = bool(input('State True if there is only 1 model for hyper-parameter optimisation (default=False) : '))
        CV = int(input('State the no of folds for CV : '))

        for i in range(0, len(self.model_obj_list)) :
            Grid_Search = GridSearchCV(estimator = self.model_obj_list[i], 
                                    param_grid = self.model_param_dict[(list(self.model_param_dict.keys()))[i]],
                                    scoring = None,
                                    n_jobs = None,
                                    cv = CV,
                                    return_train_score = True
                                    )
            grid = Grid_Search.fit(self.x_train_data, self.y_train_data)
            Hyper_param_optimized_models.append(grid.best_estimator_)

        return Hyper_param_optimized_models


    def Random_Search(self, no_iterations, cv = 5):
        # printing the docstring of RandomSearchCV, for informed user input.
        print(RandomSearchCV.__doc__)
        
        # making the return variables 'global', for further use of these in a project
        global Hyper_param_optimized_models
        Hyper_param_optimized_models = []

        # user inputs
        no_iterations = int(input('State the number of parameter settings that are sampled, it trades off runtime vs quality of the solution :'))
        scoring = str(input('State the scoring parameter or a list of scoring parameters, (default=None) :'))
        no_jobs = int(input('State the no of jobs to run in parallel, (default=None) :'))

        for i in range(0, len(self.model_obj_list)) :
            Random_Search = RandomSearchCV(estimator = self.model_obj_list[i], 
                                    param_distributions = self.model_param_dict[(list(self.model_param_dict.keys()))[i]],
                                    n_iter = no_iterations,
                                    scoring = scoring,
                                    n_jobs = no_jobs,
                                    cv = splitter,
                                    return_train_score = True,
                                    error_score = 9999
                                    )
            best_estimator = Grid_search.fit(self.x_train_data, self.y_train_data).best_estimator_
            Hyper_param_optimized_models.append(best_estimator)

        return Hyper_param_optimized_models