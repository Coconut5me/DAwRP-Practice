import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split

#%% - Init data
df = pd.read_csv('data/ACG.csv', index_col="Date", parse_dates=True)

#%% - Define the number of splits (number of folds) for time series cross-validation
n_splits = 5

#%%- Initiate TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=n_splits, test_size=30) #, gap=2)
# train_data, test_data = train_test_split(df, test_size=0.1, suffle=False) #

#%% - Iterate over the splits and split the data
for train_index, test_index in tscv.split(df):
    train_data = df.iloc[train_index]
    test_data = df.iloc[test_index]

    #Fit model, evaluation

    print("Train data length: ", len(train_data))
    print("Test data length: ", len(test_data))
    print("")