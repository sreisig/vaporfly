import pandas as pd
import numpy as np
import os 
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.stats import percentileofscore
import sys
from sklearn.metrics import r2_score

from keras.models import Model
from keras.layers import *
from keras import backend as K

"""
Loads the data from input files into a dataframe, optionally doing one hot encoding and runner binning 
"""
def load_data(input_folder, one_hot_encode=False, do_name=True, runner_bins=False):
    cols = ['match_name', 'marathon', 'time_minutes', 'vaporfly'] # not using year column 

    dtypes = {'time_minutes' : np.float64} # convert true/false to 1/0 and make sure time is a float

    files = os.listdir(input_folder)
    csvs = [f for f in files if f[-4:] == '.csv']
    df = None

    for csv in csvs:
        temp = pd.read_csv(os.path.join(input_folder, csv), usecols=cols, dtype=dtypes).dropna()

        # women have gender value of 1, men have gender value of 0
        if 'women' in csv:
            temp['gender'] = [1.0] * len(temp)
        else:
            temp['gender'] = [0.0] * len(temp)

        if df is None:
            df = temp
        else:
            df = pd.concat([df, temp])
            
    if runner_bins != False:
        # Each runner is assigned a value from 0 to runner_bins based on mean marathon times, with 0 being the fastest

        # need to first do this for male runners and female runners separately.
        df_male = df[df['gender'] == 0]
        df_female = df[df['gender'] == 1]


        runner_times = {} # dict of runner name to mean marathon time
        runner_to_bin = {} # dict of runner name to assigned bin
        for runner in np.unique(df_male['match_name'].values):
            runner_times[runner] = np.mean(df_male[df_male['match_name'] == runner]['time_minutes'].values)

        times = list(runner_times.values())
        for runner in np.unique(df_male['match_name'].values):
            perc = percentileofscore(times, runner_times[runner])

            # perc is a number between 0 and 100. Can just divide by runner_bins to get bin
            runner_to_bin[runner] = int(perc / runner_bins)

        # now do female
        runner_times = {} # reset dict
        for runner in np.unique(df_female['match_name'].values):
            runner_times[runner] = np.mean(df_female[df_female['match_name'] == runner]['time_minutes'].values)

        times = list(runner_times.values())
        for runner in np.unique(df_female['match_name'].values):
            perc = percentileofscore(times, runner_times[runner])

            # perc is a number between 0 and 100. Can just divide by runner_bins to get bin
            runner_to_bin[runner] = int(perc / runner_bins)
        

        # now runner_to_bin is populated, just assign it in original dataframe

        df['match_name'] = df['match_name'].apply(lambda x: runner_to_bin[x])

    if one_hot_encode:
        # one hot encode the predictors match_name and marathon
        one_hot_cols = ['marathon']
        prefix = ['race_']
        if do_name:
            if runner_bins == False:
                one_hot_cols.append('match_name')
                prefix.append('runner_')
            else:
                one_hot_cols.append('match_name')
                prefix.append('runner_bin_')
        else:
            df = df[[col for col in df.columns if col != 'match_name']] # get rid of match name
        df = pd.get_dummies(df, columns=one_hot_cols, prefix=prefix, prefix_sep='')

    return df


def get_data(input_folder, one_hot_encode=False, do_name=True, return_cols=False, runner_bins=10):
    df = load_data(input_folder, one_hot_encode=one_hot_encode, do_name=do_name, runner_bins=runner_bins)

    y_cols = ['time_minutes']
    x_cols = [col for col in df.columns if col not in y_cols]

    x = df[x_cols].values
    y = df[y_cols].values

    if return_cols :
        return x, y, x_cols
    return x, y



'''
Splits so that there is no runner in the test set that is not also in the train set
Rand parameter is the probability that every additional datapoint for a runner is added to the test set
    Larger rand = larger test set
'''
def get_train_test(x, y, cols, rand=0.2):
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    runners = [col for col in cols if 'runner_' in col]
    
    runner_counts = {runner:0 for runner in runners}
    
    for i in range(len(x)):
        datapoint = x[i]
        runner = [c for c in [cols[i] for i in range(len(datapoint)) if datapoint[i] == 1] if 'runner_' in c][0]
        runner_counts[runner] += 1

        if runner_counts[runner] > 1 and random.random() < rand:
            x_test.append(x[i])
            y_test.append(y[i])
        else:
            x_train.append(x[i])
            y_train.append(y[i])
    
    print(len(x_train))
    print(len(x_test))

    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)


'''
gets inputs to the additive network. Shoe effects are the columns to be used for the shoe effect branch.
Also allowed to be a mask (such as 'runner_') to select all columns matching the mask

'''
def get_additive_inputs(x, cols, shoe_effects=[], get_cols=False):
    x_df = pd.DataFrame(x, columns=cols) # easier to select columns using a dataframe

    x_normal = x_df[[col for col in cols if col != 'vaporfly']]

    x_shoe = x_df['vaporfly'] # this input activates the shoe effect branch

    shoe_effect_columns = ['vaporfly'] # the inputs to make the shoe effect branch

    for eff in shoe_effects:
        if eff == 'gender':
            shoe_effect_columns.append(eff)
        else:
            for new_col in [col for col in cols if eff in col]:
                shoe_effect_columns.append(new_col)
    
    x_shoe_effects = x_df[shoe_effect_columns]

    if not get_cols:
        return [x_normal.values, x_shoe_effects.values, x_shoe.values]
    else:
        return [x_normal.values, x_shoe_effects.values, x_shoe.values], [[col for col in cols if col != 'vaporfly'], shoe_effect_columns, ['vaporfly']]

'''
returns just the inputs where vaporfly is true
(or false, if inverse=True)
'''
def get_vaporfly_inputs(x, y, inverse=False):
    # x is list [normal inputs, shoe_effect_inputs, shoe_input]
    if inverse == False:
        indices = np.where(x[2] == 1)
    else:
        indices = np.where(x[2] == 0)

    x_ret = [x[0][indices], x[1][indices], x[2][indices]]
    y_ret = y[indices]

    return x_ret, y_ret


def split_gender(x, y, cols):
    indices_f = [i for i in range(len(x[0])) if x[0][i][cols[0].index('gender')] == 1]
    indices_m = [i for i in range(len(x[0])) if x[0][i][cols[0].index('gender')] == 0]
    # print(indices_f)
    # print(indices_m)

    x_m = [x[0][indices_m], x[1][indices_m], x[2][indices_m]]
    y_m = y[indices_m]

    x_f = [x[0][indices_f], x[1][indices_f], x[2][indices_f]]
    y_f = y[indices_f]

    return x_m, x_f, y_m, y_f

def get_feedforward_model(x):
    # right now, use network with 4 hidden layers with 50 nodes each
    input_layer = Input(shape=(len(x[0]),))
    d = Dropout(0.1)(input_layer)
    dense = Dense(50, activation='relu')(d)
    dense = Dense(50, activation='relu')(dense)
    dense = Dense(50, activation='relu')(dense)
    dense = Dense(50, activation='relu')(dense)
    outlayer = Dense(1, activation='linear')(dense)
    model = Model(inputs=input_layer, outputs=outlayer)
    
    # print(model.summary())
    return model

'''
gets the additive network. Shoe effects are the columns to be used for the shoe effect branch.
Also allowed to be a mask (such as 'runner_') to select all columns matching the mask

'''
def get_additive_model(cols, shoe_effects=[]):

    shoe_effect_columns = ['vaporfly']
    for eff in shoe_effects:
        if eff == 'gender':
            shoe_effect_columns.append(eff)
        else:
            for new_col in [col for col in cols if eff in col]:
                shoe_effect_columns.append(new_col)

    inlayer_normal = Input(shape=(len(cols) - 1, )) #  everything but vaporfly
    inlayer_shoe_effect = Input(shape=(len(shoe_effect_columns),)) 
    inlayer_shoe = Input(shape=(1,))

    # make normal network
    d = Dropout(0.1)(inlayer_normal)
    temp = Dense(50, activation='relu')(d)
    temp = Dense(50, activation='relu')(temp)
    temp = Dense(50, activation='relu')(temp)
    temp = Dense(50, activation='relu')(temp)
    temp = Dense(50, activation='relu')(temp)
    temp = Dense(50, activation='relu')(temp)
    temp = Dense(20, activation='relu')(temp)
    normal_out = Dense(1, activation='linear')(temp)

    # make shoe effect network
    temp = Dense(50, activation='relu')(inlayer_shoe_effect)
    temp = Dense(20, activation='relu')(temp)
    shoe_effect_out = Dense(1, activation='linear')(temp)

    # activate the shoe effect network based on the inlayer shoe input (if vaporfly is present)
    shoe_out = Multiply()([shoe_effect_out, inlayer_shoe])

    # add shoe effect with normal regression
    outlayer = Add()([normal_out, shoe_out])

    model = Model(inputs=[inlayer_normal, inlayer_shoe_effect, inlayer_shoe], outputs=outlayer)
    # print(model.summary())

    # to get the activations for the additive layers
    layer_model = Model(inputs=[inlayer_normal, inlayer_shoe_effect, inlayer_shoe], outputs=[normal_out, shoe_out])

    return model, layer_model