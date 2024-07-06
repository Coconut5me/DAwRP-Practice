#%%
import pandas as pd
from pmdarima import ARIMA
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
import warnings
warnings.filterwarnings("ignore")

#%%
def rolling_window(df_train, n_folds, split_train_rate):
    df_fold_rolling_window = pd.DataFrame(columns=['Fold', 'start_index', 'train_length', 'test_length', 'end_index'])

    train_list=[]
    test_list=[]

    # Calculate window size
    window_size = len(df_train) // n_folds

    for i in range(n_folds):
        start_index = i * window_size
        end_index = min((i + 1) * window_size, len(df_train))
        window_data = df_train[start_index:end_index]

        # Split the window data into training and validation sets
        train_data, test_data = window_data[:int(len(window_data) * split_train_rate)], window_data[int(len(window_data) * split_train_rate):]
        train_list.append(train_data)
        test_list.append(test_data)

        # Tạo một Series chứa thông tin của fold hiện tại
        fold_info = pd.Series({
            'Fold': f'Fold {i + 1}',
            'start_index': start_index,
            'train_length': len(train_data),
            'test_length': len(test_data),
            'end_index': end_index
        })

        # Thêm thông tin của fold vào DataFrame
        df_fold_rolling_window = df_fold_rolling_window._append(fold_info, ignore_index=True)

    return train_list, test_list, df_fold_rolling_window

#%%
def expanding_window(df_train, n_folds, split_train_rate):
    df_fold_expanding_window = pd.DataFrame(columns=['Fold', 'start_index', 'train_length', 'test_length', 'end_index'])

    train_list=[]
    test_list=[]
    start_index = 0
    window_size = len(df_train) // n_folds
    end_index = window_size

    for i in range(n_folds):
        # Extract data within the window
        window_data = df_train[:end_index]

        # Split the window data into training and testing sets
        train_data, test_data = window_data[:int(len(window_data) * split_train_rate)], window_data[int(len(window_data) * split_train_rate):]
        train_list.append(train_data)
        test_list.append(test_data)

        # Create a Series containing information about the current fold
        fold_info = pd.Series({
            'Fold': f'Fold {i + 1}',
            'start_index': start_index,
            'train_length': len(train_data),
            'test_length': len(test_data),
            'end_index': end_index
        })

        # Add fold information to the DataFrame
        df_fold_expanding_window = df_fold_expanding_window._append(fold_info, ignore_index=True)

        # Increment the indices for the next window
        start_index = 0
        end_index += window_size

    return train_list, test_list, df_fold_expanding_window

#%% - ARIMA
def fit_arima(train_data):

    # ARIMA model fitting
    stepwise_fit = auto_arima(train_data, trace=True, suppress_warnings=True)
    model = ARIMA(train_data, order=stepwise_fit.order, trend='t')
    fitted = model.fit()

    return fitted
