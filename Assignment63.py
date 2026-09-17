import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import StandardScaler


################################################################################
# 1 : LOAD DATASET
################################################################################
df= pd.read_csv("Customer_Loan_Approval.csv")

print("shape of data",df.shape)
print("Data ; ",df.head())

################################################################################
# 2 : MISSING VALUES 
################################################################################

print("\nMissing values in data :\n",df.isnull().sum())

################################################################################
# 3 : SEPRATE DATA
################################################################################
X= df.drop("LoanApproved",axis=1)
Y= df["LoanApproved"]

################################################################################
# 4 : TRAIN TEST SPLIT 
################################################################################
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

################################################################################
# 5 : FEATURE SCALING 
################################################################################
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

################################################################################
# 6 : HYPERPARAMETER EXPERIMENT1 -ACTIVATION 
################################################################################

acitvation=['identity', 'logistic', 'tanh', 'relu']

print("Activation Experiment :")

for act in acitvation:
    model = MLPClassifier(
        hidden_layer_sizes=(32,16),
        activation=act,
        solver='adam',
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f"Activation: {act}, Accuracy: {accuracy:.4f}")



################################################################################
# 7 : HYPERPARAMETER EXPERIMENT2 - HIDDEN LAYER 
################################################################################

hidden_layer_sizes = [
    (10),
    (20,10),
    (50,25),
    (100,50,25)
]

print("\nHidden Layer Sizes Experiment :")

for layers in hidden_layer_sizes:
    model = MLPClassifier(
        hidden_layer_sizes=layers,
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f"Hidden Layer Sizes: {layers}, Accuracy: {accuracy:.4f}")


################################################################################
# 8 : HYPERPARAMETER EXPERIMENT 3- LEARNING RATE   
################################################################################

learning_rates = [0.0001, 0.001, 0.01,0.1]

print("\nLearning Rate Experiment :")

for lr in learning_rates:
    model = MLPClassifier(
        hidden_layer_sizes=(32,16),
        activation='relu',
        solver='adam',
        learning_rate_init=lr,
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f"Learning Rate: {lr}, Accuracy: {accuracy:.4f}")