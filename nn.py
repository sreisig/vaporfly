# Combined M/F model

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
from keras.callbacks import EarlyStopping

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



def compare_models(test_size=0.2):

    input_folder = 'data'

    x, y, cols = get_data(input_folder, one_hot_encode=True, do_name=True, return_cols=True)
    # print (cols)

    if test_size != 0:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=True)
    else:
        x_train = x
        y_train = y
    # x_train, x_test, y_train, y_test = get_train_test(x, y, cols)

    linear_model = LinearRegression()
    linear_model.fit(x_train, y_train)

    linear_train_r2 = linear_model.score(x_train, y_train)
    if test_size != 0:
        linear_test_r2 = linear_model.score(x_test, y_test)

    nn = get_feedforward_model(x_train)
    nn.compile(optimizer='adam', loss='mean_squared_error')

    if test_size != 0:
        hist = nn.fit(x=x_train, y=y_train, batch_size = 32, epochs=150, validation_data = (x_test, y_test), callbacks=[], verbose=0)
    else:
        hist = nn.fit(x=x_train, y=y_train, batch_size = 32, epochs=150, callbacks=[], verbose=0)

    nn_train_r2 = r2_score(y_train, nn.predict(x_train))
    if test_size != 0:
        nn_test_r2 = r2_score(y_test, nn.predict(x_test))


    # now do additive network
    shoe_effects = ['gender', 'race_']
    an, activation_network = get_additive_model(cols, shoe_effects=shoe_effects)
    # construct input vectors
    x_train_an, new_cols = get_additive_inputs(x_train, cols, shoe_effects=shoe_effects, get_cols=True)
    if test_size != 0:
        x_test_an = get_additive_inputs(x_test, cols, shoe_effects=shoe_effects)

    an.compile(optimizer='adam', loss='mean_squared_error')

    if test_size != 0:
        hist = an.fit(x=x_train_an, y=y_train, batch_size = 32, epochs=200, validation_data = (x_test_an, y_test), callbacks=[], verbose=0)
    else:
        hist = an.fit(x=x_train_an, y=y_train, batch_size = 32, epochs=200, callbacks=[], verbose=0)

    an_train_r2 = r2_score(y_train, an.predict(x_train_an))
    if test_size != 0:
        an_test_r2 = r2_score(y_test, an.predict(x_test_an))


    # get activations for vaporfly and non vaporfly
    # not really sure if i should be doing this on train data, test data, or both
    vap_x, vap_y = get_vaporfly_inputs(x_train_an, y_train)
    nonvap_x, nonvap_y = get_vaporfly_inputs(x_train_an, y_train, inverse=True)
    print (vap_x)
    print (nonvap_x)
    print (np.mean(vap_y))
    print (np.mean(nonvap_y))

    # need to also divide by gender
    vap_x_m, vap_x_f, vap_y_m, vap_y_f = split_gender(vap_x, vap_y, new_cols)
    nonvap_x_m, nonvap_x_f, nonvap_y_m, nonvap_y_f = split_gender(nonvap_x, nonvap_y, new_cols)

    # print([nonvap_x_f[0][i][new_cols[0].index('gender')] for i in range(len(nonvap_x_f[0]))])

    preds_vap_m = activation_network.predict(vap_x_m)
    preds_nonvap_m = activation_network.predict(nonvap_x_m)
    preds_total_vap_m = an.predict(vap_x_m)
    preds_total_nonvap_m = an.predict(nonvap_x_m)

    preds_vap_f = activation_network.predict(vap_x_f)
    preds_nonvap_f = activation_network.predict(nonvap_x_f)
    preds_total_vap_f = an.predict(vap_x_f)
    preds_total_nonvap_f = an.predict(nonvap_x_f)   


    # print(cols)
    print("Linear Model:")
    print("\tTrain R Squared: " + str(linear_train_r2))
    if test_size != 0:
        print("\tTest R Squared: " + str(linear_test_r2))

    print("Feedforward NN Model:")
    print("\tTrain R Squared: " + str(nn_train_r2))
    if test_size != 0:
        print("\tTest R Squared: " + str(nn_test_r2))


    print("Additive Network Model:")
    print("\tShoe Effects: " + str(shoe_effects))
    print("\tTrain R Squared: " + str(an_train_r2))
    if test_size != 0:
        print("\tTest R Squared: " + str(an_test_r2))

    print("Male")
    print("\tActivations Vaporfly:")
    print("\t\tNormal: mean=" + str(np.mean(preds_vap_m[0])) + ", std=" + str(np.std(preds_vap_m[0])))
    print("\t\tShoe: mean=" + str(np.mean(preds_vap_m[1])) + ", std=" + str(np.std(preds_vap_m[1])))
    print("\t\tTotal: mean=" + str(np.mean(preds_total_vap_m)) + ", std=" + str(np.std(preds_total_vap_m)))
    print("\tActivations Non Vaporfly:")
    print("\t\tNormal: mean=" + str(np.mean(preds_nonvap_m[0])) + ", std=" + str(np.std(preds_nonvap_m[0])))
    print("\t\tShoe: mean=" + str(np.mean(preds_nonvap_m[1])) + ", std=" + str(np.std(preds_nonvap_m[1])))
    print("\t\tTotal: mean=" + str(np.mean(preds_total_nonvap_m)) + ", std=" + str(np.std(preds_total_nonvap_m)))

    print("Female")
    print("\tActivations Vaporfly:")
    print("\t\tNormal: mean=" + str(np.mean(preds_vap_f[0])) + ", std=" + str(np.std(preds_vap_f[0])))
    print("\t\tShoe: mean=" + str(np.mean(preds_vap_f[1])) + ", std=" + str(np.std(preds_vap_f[1])))
    print("\t\tTotal: mean=" + str(np.mean(preds_total_vap_f)) + ", std=" + str(np.std(preds_total_vap_f)))
    print("\tActivations Non Vaporfly:")
    print("\t\tNormal: mean=" + str(np.mean(preds_nonvap_f[0])) + ", std=" + str(np.std(preds_nonvap_f[0])))
    print("\t\tShoe: mean=" + str(np.mean(preds_nonvap_f[1])) + ", std=" + str(np.std(preds_nonvap_f[1])))
    print("\t\tTotal: mean=" + str(np.mean(preds_total_nonvap_f)) + ", std=" + str(np.std(preds_total_nonvap_f)))


    print("Actual times vaporfly:")
    print("\tTotal: mean=" + str(np.mean(vap_y)) + ", std=" + str(np.std(vap_y)))
    print("\tMale: mean=" + str(np.mean(vap_y_m)) + ", std=" + str(np.std(vap_y_m)))
    print("\tFemale: mean=" + str(np.mean(vap_y_f)) + ", std=" + str(np.std(vap_y_f)))

    print("Actual times nonvaporfly:")
    print("\tTotal: mean=" + str(np.mean(nonvap_y)) + ", std=" + str(np.std(nonvap_y)))
    print("\tMale: mean=" + str(np.mean(nonvap_y_m)) + ", std=" + str(np.std(nonvap_y_m)))
    print("\tFemale: mean=" + str(np.mean(nonvap_y_f)) + ", std=" + str(np.std(nonvap_y_f)))
    






if __name__ == '__main__':
    compare_models(test_size=0.0)