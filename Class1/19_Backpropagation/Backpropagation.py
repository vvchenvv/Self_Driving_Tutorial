
# coding: utf-8

# In[1]:


import numpy as np


def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))


x = np.array([0.5, 0.1, -0.2])
target = 0.6
learnrate = 0.5

weights_input_hidden = np.array([[0.5, -0.6],
                                 [0.1, -0.2],
                                 [0.1, 0.7]])

weights_hidden_output = np.array([0.1, -0.3])

## Forward pass
hidden_layer_input = np.dot(x, weights_input_hidden)
hidden_layer_output = sigmoid(hidden_layer_input)

output_layer_in = np.dot(hidden_layer_output, weights_hidden_output)
output = sigmoid(output_layer_in)

## Backwards pass
## TODO: Calculate error
error = target - output

# TODO: Calculate error gradient for output layer
del_err_output = error * output * (1 - output)

# TODO: Calculate error gradient for hidden layer
del_err_hidden = np.dot(del_err_output, weights_hidden_output) *                  hidden_layer_output * (1 - hidden_layer_output)

# TODO: Calculate change in weights for hidden layer to output layer
delta_w_h_o = learnrate * del_err_output * hidden_layer_output

# TODO: Calculate change in weights for input layer to hidden layer
delta_w_i_h = learnrate * del_err_hidden * x[:, None]

print('Change in weights for hidden layer to output layer:')
print(delta_w_h_o)
print('Change in weights for input layer to hidden layer:')
print(delta_w_i_h)


# In[2]:


import numpy as np
import pandas as pd

data=pd.read_csv("binary.csv")
data['rank'].astype(int)

def rank1_std(x):
    if x['rank'] == 1:
        return 1
    else:
        return 0
def rank2_std(x):
    if x['rank'] == 2:
        return 1
    else:
        return 0
def rank3_std(x):
    if x['rank'] == 3:
        return 1
    else:
        return 0
def rank4_std(x):
    if x['rank'] == 4:
        return 1
    else:
        return 0
    
def admit_std(x):
    if x['admit']:
        return True
    else:
        return False

data['rank_1']=data.apply(rank1_std,axis=1)
data['rank_2']=data.apply(rank2_std,axis=1)
data['rank_3']=data.apply(rank3_std,axis=1)
data['rank_4']=data.apply(rank4_std,axis=1)
data['admit']=data.apply(admit_std,axis=1)

gre_mean=data['gre'].mean()
gre_max=data['gre'].max()
gre_min=data['gre'].min()
gre_std=data['gre'].std()

gpa_mean=data['gpa'].mean()
gpa_max=data['gpa'].max()
gpa_min=data['gpa'].min()
gpa_std=data['gpa'].std()

data['gre']=data['gre'].map(lambda x: (x-gre_mean)/gre_std)
data['gpa']=data['gpa'].map(lambda x: (x-gpa_mean)/gpa_std)
del data['rank']
data.head(20)


# In[1]:


import numpy as np
import pandas as pd

admissions = pd.read_csv("binary.csv")

# Make dummy variables for rank
data = pd.concat([admissions, pd.get_dummies(admissions['rank'], prefix='rank')], axis=1)
data = data.drop('rank', axis=1)

# Standarize features
for field in ['gre', 'gpa']:
    mean, std = data[field].mean(), data[field].std()
    data.loc[:,field] = (data[field]-mean)/std
    
# Split off random 10% of the data for testing
np.random.seed(21)
sample = np.random.choice(data.index, size=int(len(data)*0.9), replace=False)
data, test_data = data.ix[sample], data.drop(sample)

# Split into features and targets
features, targets = data.drop('admit', axis=1), data['admit']
features_test, targets_test = test_data.drop('admit', axis=1), test_data['admit']


# In[3]:


# features=data[['gre','gpa','rank_1','rank_2','rank_3','rank_4']][:390]
# targets=data['admit'][:390]
# features_test=data[['gre','gpa','rank_1','rank_2','rank_3','rank_4']][390:]
# targets_test=data['admit'][390:]
# features.head(10)

np.random.seed(21)

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))


# Hyperparameters
n_hidden = 2  # number of hidden units
epochs = 900
learnrate = 0.005

n_records, n_features = features.shape
last_loss = None
# Initialize weights
weights_input_hidden = np.random.normal(scale=1 / n_features ** .5,
                                        size=(n_features, n_hidden))
weights_hidden_output = np.random.normal(scale=1 / n_features ** .5,
                                         size=n_hidden)

for e in range(epochs):
    del_w_input_hidden = np.zeros(weights_input_hidden.shape)
    del_w_hidden_output = np.zeros(weights_hidden_output.shape)
    for x, y in zip(features.values, targets):
        ## Forward pass ##
        # TODO: Calculate the output
        hidden_input = np.dot(x, weights_input_hidden)
        hidden_output = sigmoid(hidden_input)

        output = sigmoid(np.dot(hidden_output,
                                weights_hidden_output))

        ## Backward pass ##
        # TODO: Calculate the error
        error = y - output

        # TODO: Calculate error gradient in output unit
        output_error = error * output * (1 - output)

        # TODO: propagate errors to hidden layer
        hidden_error = np.dot(output_error, weights_hidden_output) *                        hidden_output * (1 - hidden_output)

        # TODO: Update the change in weights
        del_w_hidden_output += output_error * hidden_output
        del_w_input_hidden += hidden_error * x[:, None]

    # TODO: Update weights
    weights_input_hidden += learnrate * del_w_input_hidden / n_records
    weights_hidden_output += learnrate * del_w_hidden_output / n_records

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        hidden_output = sigmoid(np.dot(x, weights_input_hidden))
        out = sigmoid(np.dot(hidden_output,
                             weights_hidden_output))
        loss = np.mean((out - targets) ** 2)

        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss

# Calculate accuracy on test data
hidden = sigmoid(np.dot(features_test, weights_input_hidden))
out = sigmoid(np.dot(hidden, weights_hidden_output))
predictions = out > 0.5
print(out)
print(predictions)
print(targets_test)
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))


# In[13]:


# for x, y in zip(features.values, targets):
#     print(x)
#     print(y)
print(True-0.5)
print(False-0.5)

