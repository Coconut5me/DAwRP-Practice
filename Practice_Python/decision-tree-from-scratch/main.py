#%% - Import Lib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from tree import DecisionTree


def label_encode(column_name, data):
    label_encoder = LabelEncoder()
    return label_encoder.fit_transform(data[column_name])


#%% - Load data
data_frame = pd.read_csv("Buys_computer.csv")

X = data_frame.drop("Buys_computer", axis=1)

X["Encoded_Age"] = X["Age"].map({"<=30": 0, "31..40": 1, ">40": 2})
X["Encoded_Income"] = X["Income"].map({"low": 0, "medium": 1, "high": 2})
X["Encoded_Student"] = label_encode("Student", X)
X["Encoded_Credit_rating"] = label_encode("Credit_rating", X)
X.drop(["Age", "Income", "Student", "Credit_rating"], axis=1, inplace=True)
y = label_encode("Buys_computer", data_frame)

#%%
tree = DecisionTree(max_depth=10)
print(X)
tree.fit(X.values, y)
tree.print_tree()
