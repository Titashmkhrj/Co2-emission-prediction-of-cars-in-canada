class Model_Selection :
	'''
	Class documentation.
	'''

	def __init__ (self, 
				x_train_data,
				y_train_data, 
				Optimized_models
				):
		self.x_train_data = x_train_data
		self.y_train_data = y_train_data
		self.Optimized_models = Optimized_models



	def CV_score(self, scoring = None, cv = 5, no_jobs = None):
        # printing the docstring of cross_validate, for informed user input.
        print(cross_validate.__doc__)
        
        # making the return variables 'global', for further use of these in a project
        global 

        # user inputs
        scoring = str(input('State the scoring parameter or a list of scoring parameters, (default=None) :'))
        no_jobs = int(input('State the no of jobs to run in parallel, (default=None) :'))

        final_scores = pd.DataFrame()
        for i in range(0, len(Optimized_models)):
        	scores = cross_validate(estimator = self.Optimized_models[i],
    								X = self.x_train_data,
    								y = self.y_train_data,
    								scoring = scoring,
    								cv = splitter,
    								n_jobs = no_jobs,
    								)
            scores_df = pd.DataFrame(scores)
            final_scores = pd.concat([scores_df, final_scores])

        return final_scores