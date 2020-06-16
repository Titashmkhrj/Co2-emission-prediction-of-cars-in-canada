from sklearn.model_selection import (KFold,
                                     StratifiedKFold,
                                     GroupKFold,
                                     RepeatedKFold,
                                     RepeatedStratifiedKFold,
                                     ShuffleSplit,
                                     GroupShuffleSplit,
                                     StratifiedShuffleSplit,
                                     LeaveOneOut,
                                     LeaveOneGroupOut,
                                     LeavePGroupsOut,
                                     LeavePOut,
                                     TimeSeriesSplit)



def Splitter (no_of_repeats = 10, test_size = None, train_size = None, no_of_groups = 2, p = 2, max_train_size = None):
    #def __init__(TITASH,no_of_splits = 5, rand_state = None, splitter_choice = 0, no_of_repeats = 10, test_size = None, train_size = None ):
    splitter_choice = int(input('''State the name of the splitter, by stating the representative index from the following options :
    * KFold >>>>>>>>>>>>>>>>>>>> 0
    * StratifiedKFold >>>>>>>>>> 1
    * GroupKFold >>>>>>>>>>>>>>> 2
    * RepeatedKFold >>>>>>>>>>>> 3
    * RepeatedStratifiedKFold >> 4
    * ShuffleSplit >>>>>>>>>>>>> 5
    * GroupShuffleSplit >>>>>>>> 6
    * StratifiedShuffleSplit >>> 7
    * LeaveOneOut >>>>>>>>>>>>>> 8
    * LeaveOneGroupOut >>>>>>>>> 9
    * LeavePGroupsOut >>>>>>>>>> 10
    * LeavePOut >>>>>>>>>>>>>>>> 11
    * TimeSeriesSplit >>>>>>>>>> 12'''))
    no_of_splits = int(input('State the number of splits :'))
    rand_state = int(input('State the random sate, by default it is 0 :'))
    shuffle = bool(input('State yes for shuffle to be True, and absent input implies False :'))
    if splitter_choice == 3 or splitter_choice == 4:
        no_of_repeats = int(input('State the number of times the cross-validator needs to be repeated for RepeatedKFold or RepeatedStratifiedKFold :'))
    elif splitter_choice == 5 or splitter_choice == 6 or splitter_choice == 7:
        test_size = float(input('''For ShuffleSplit or GroupsShuffleSplit or StratifiedShuffleSplit,
        state the representative proportion (should be between 0.0 and 1.0) of the dataframe to include in the test split :'''))
        train_size = float(input('''For ShuffleSplit or GroupShuffleSplit or StratifiedShuffleSplit,
        state the representative proportion (should be between 0.0 and 1.0) of the dataframe to include in the train split :'''))
    elif splitter_choice == 10:
        no_of_groups = int(input('For LeavePGroupsout, state the number of groups (p) to leave out in the test split :'))
    elif splitter_choice == 11:
        p = int(input('For LeavePOut, state the size of the test sets. Must be strictly less than the number of samples :'))
    elif splitter_choice == 12:
        max_train_size = int(input('For using TimeSeriesSplit, state the maximum size for a single training set :'))
                
    
    global splitter
    
    splitter_dict = {0 : KFold(n_splits = no_of_splits, random_state = rand_state, shuffle = shuffle),
                    1 : StratifiedKFold(n_splits = no_of_splits, random_state = rand_state, shuffle = shuffle),
                    2 : GroupKFold(n_splits = no_of_splits),
                    3 : RepeatedKFold(n_splits = no_of_splits, n_repeats = no_of_repeats, random_state = rand_state),
                    4 : RepeatedStratifiedKFold(n_splits = no_of_splits, n_repeats = no_of_repeats, random_state = rand_state),
                    5 : ShuffleSplit(n_splits = no_of_splits, random_state = rand_state, test_size = test_size, train_size = train_size),
                    6 : GroupShuffleSplit(n_splits = no_of_splits, random_state = rand_state, test_size = test_size, train_size = train_size),
                    7 : StratifiedShuffleSplit(n_splits = no_of_splits, random_state = rand_state, test_size = test_size, train_size = train_size),
                    8 : LeaveOneOut(),
                    9 : LeaveOneGroupOut(),
                    10 : LeavePGroupsOut(n_groups = no_of_groups),
                    11 : LeavePOut(p = p),
                    12 : TimeSeriesSplit(n_splits = no_of_splits, max_train_size = max_train_size)}
    
    if splitter_choice == 0:
        splitter = splitter_dict[splitter_choice]
    elif splitter_choice == 1:
        splitter = splitter_dict[splitter_choice]
    elif splitter_choice == 2:
        splitter = splitter_dict[splitter_choice]
    elif splitter_choice == 3:
        splitter = splitter_dict[splitter_choice]
    elif splitter_choice == 4:
        splitter = splitter_dict[splitter_choice]
    elif splitter_choice == 5:
        splitter = splitter_dict[splitter_choice]
    elif splitter_choice == 6:
        splitter = splitter_dict[splitter_choice]
    elif splitter_choice == 7:
        splitter = splitter_dict[splitter_choice]
    elif splitter_choice == 8:
        splitter = splitter_dict[splitter_choice]
    elif splitter_choice == 9:
        splitter = splitter_dict[splitter_choice]
    elif splitter_choice == 10:
        splitter = splitter_dict[splitter_choice]
    elif splitter_choice == 11:
        splitter = splitter_dict[splitter_choice]
    elif splitter_choice == 12:
        splitter = splitter_dict[splitter_choice]    
    return splitter