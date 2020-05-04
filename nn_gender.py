# Separate M/F models

import pandas as pd
import numpy as np
import os 
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sys

from keras.models import Model
from keras.layers import *
from keras import backend as K
from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from scipy.stats import percentileofscore


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
    
    # print(len(x_train))
    # print(len(x_test))

    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)


'''
gets inputs to the additive network. Shoe effects are the columns to be used for the shoe effect branch.
Also allowed to be a mask (such as 'runner_') to select all columns matching the mask

'''
def get_additive_inputs(x, cols, shoe_effects=[]):
    x_df = pd.DataFrame(x, columns=cols) # easier to select columns using a dataframe

    x_normal = x_df[[col for col in cols if col != 'vaporfly']].values

    x_shoe = x_df['vaporfly'].values # this input activates the shoe effect branch

    shoe_effect_columns = ['vaporfly'] # the inputs to make the shoe effect branch

    for eff in shoe_effects:
        if eff == 'gender':
            shoe_effect_columns.append(eff)
        else:
            for new_col in [col for col in cols if eff in col]:
                shoe_effect_columns.append(new_col)
    
    x_shoe_effects = x_df[shoe_effect_columns].values

    return [x_normal, x_shoe_effects, x_shoe]

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


def split_gender(x, cols, y=None):
    indices_f = [i for i in range(len(x[0])) if x[0][i][cols.index('gender') -1] == 1]
    indices_m = [i for i in range(len(x[0])) if x[0][i][cols.index('gender') -1] == 0]
    # print(indices_f)
    # print(indices_m)

    x_m = [x[0][indices_m], x[1][indices_m], x[2][indices_m]]
    if y is not None:
        y_m = y[indices_m]

    x_f = [x[0][indices_f], x[1][indices_f], x[2][indices_f]]
    if y is not None:
        y_f = y[indices_f]

    if y is not None:
        return x_m, x_f, y_m, y_f
    else:
        return x_m, x_f

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


def compare_models(test_size=0.2, shoe_effects=['gender', 'race_']):
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
    
    # early stopping to prevent overfitting
    if test_size != 0:
        hist = nn.fit(x=x_train, y=y_train, batch_size = 32, epochs=150, validation_data = (x_test, y_test), callbacks=[], verbose=0)
    else:
        hist = nn.fit(x=x_train, y=y_train, batch_size = 32, epochs=150, callbacks=[], verbose=0)

    nn_train_r2 = r2_score(y_train, nn.predict(x_train))
    if test_size != 0:
        nn_test_r2 = r2_score(y_test, nn.predict(x_test))


    # now do additive network, one for each male and female
    an_m, activation_network_m = get_additive_model(cols, shoe_effects=shoe_effects)
    an_f, activation_network_f = get_additive_model(cols, shoe_effects=shoe_effects)


    # construct input vectors
    male_indices = [i for i in range(len(x)) if x[i][cols.index('gender')] == 0]
    female_indices = [i for i in range(len(x)) if x[i][cols.index('gender')] == 1]

    x_m = x[male_indices]
    y_m = y[male_indices]
    x_f = x[female_indices]
    y_f = y[female_indices]

    # x_train_m, x_test_m, y_train_m, y_test_m = train_test_split(x_m, y_m, test_size=0.05, shuffle=True)
    # x_train_f, x_test_f, y_train_f, y_test_f = train_test_split(x_f, y_f, test_size=0.05, shuffle=True)
    if test_size != 0:
        x_train_m, x_test_m, y_train_m, y_test_m = get_train_test(x_m, y_m, cols, rand=test_size)
        x_train_f, x_test_f, y_train_f, y_test_f = get_train_test(x_f, y_f, cols, rand=test_size)
    else:
        x_train_m = x_m
        y_train_m = y_m
        x_train_f = x_f
        y_train_f = y_f


    x_train_m = get_additive_inputs(x_train_m, cols, shoe_effects=shoe_effects)
    x_train_f = get_additive_inputs(x_train_f, cols, shoe_effects=shoe_effects)
    if test_size != 0:
        x_test_m = get_additive_inputs(x_test_m, cols, shoe_effects=shoe_effects)
        x_test_f = get_additive_inputs(x_test_f, cols, shoe_effects=shoe_effects)

    # train male
    an_m.compile(optimizer='adam', loss='mean_squared_error')

    if test_size != 0:
        hist = an_m.fit(x=x_train_m, y=y_train_m, batch_size = 32, epochs=200, validation_data = (x_test_m, y_test_m), callbacks=[], verbose=0)
    else:
        hist = an_m.fit(x=x_train_m, y=y_train_m, batch_size = 32, epochs=200, callbacks=[], verbose=0)

    an_train_r2_m = r2_score(y_train_m, an_m.predict(x_train_m))
    if test_size != 0:
        an_test_r2_m = r2_score(y_test_m, an_m.predict(x_test_m))

    # train female
    an_f.compile(optimizer='adam', loss='mean_squared_error')

    if test_size != 0:
        hist = an_f.fit(x=x_train_f, y=y_train_f, batch_size = 32, epochs=200, validation_data = (x_test_f, y_test_f), callbacks=[], verbose=0)
    else:
        hist = an_f.fit(x=x_train_f, y=y_train_f, batch_size = 32, epochs=200, callbacks=[], verbose=0)

    an_train_r2_f = r2_score(y_train_f, an_f.predict(x_train_f))
    if test_size != 0:
        an_test_r2_f = r2_score(y_test_f, an_f.predict(x_test_f))


    # get activations for vaporfly and non vaporfly
    # not really sure if i should be doing this on train data, test data, or both
    vap_x_m, vap_y_m = get_vaporfly_inputs(x_train_m, y_train_m)
    nonvap_x_m, nonvap_y_m = get_vaporfly_inputs(x_train_m, y_train_m, inverse=True)

    vap_x_f, vap_y_f = get_vaporfly_inputs(x_train_f, y_train_f)
    # print(len(vap_y_f))
    nonvap_x_f, nonvap_y_f = get_vaporfly_inputs(x_train_f, y_train_f, inverse=True)


    preds_vap_m = activation_network_m.predict(vap_x_m)
    preds_nonvap_m = activation_network_m.predict(nonvap_x_m)
    preds_total_vap_m = an_m.predict(vap_x_m)
    preds_total_nonvap_m = an_m.predict(nonvap_x_m)

    preds_vap_f = activation_network_f.predict(vap_x_f)
    preds_nonvap_f = activation_network_f.predict(nonvap_x_f)
    preds_total_vap_f = an_f.predict(vap_x_f)
    preds_total_nonvap_f = an_f.predict(nonvap_x_f)   

    y_train_vap = np.concatenate([vap_y_m, vap_y_f])
    y_train_nonvap = np.concatenate([nonvap_y_m, nonvap_y_f])

    all_preds_m = an_m.predict(x_train_m)

    # plt.scatter(y_train_m, all_preds_m)
    # plt.xlabel("Y True")
    # plt.ylabel("Y pred")
    # plt.show()

    # verify_r2 = r2_score(y_train_m, all_preds_m)

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
    if test_size != 0:
        print("\tR Squared Male: train=" + str(an_train_r2_m) + ', test=' + str(an_test_r2_m))
        print("\tR Squared Female: train=" + str(an_train_r2_f) + ', test=' + str(an_test_r2_f))
    else:
        print("\tR Squared Male: train=" + str(an_train_r2_m))
        print("\tR Squared Female: train=" + str(an_train_r2_f))        

    print("Male")
    if test_size != 0:
        print("\tCounts: Train=" + str(len(y_train_m)) + ", Test=" + str(len(y_test_m)))
    else:
        print("\tCounts: Train=" + str(len(y_train_m)))

    print("\tActivations Vaporfly:")
    print("\t\tNormal: mean=" + str(np.mean(preds_vap_m[0])) + ", std=" + str(np.std(preds_vap_m[0])))
    print("\t\tShoe: mean=" + str(np.mean(preds_vap_m[1])) + ", std=" + str(np.std(preds_vap_m[1])))
    print("\t\tTotal: mean=" + str(np.mean(preds_total_vap_m)) + ", std=" + str(np.std(preds_total_vap_m)))
    print("\tActivations Non Vaporfly:")
    print("\t\tNormal: mean=" + str(np.mean(preds_nonvap_m[0])) + ", std=" + str(np.std(preds_nonvap_m[0])))
    print("\t\tShoe: mean=" + str(np.mean(preds_nonvap_m[1])) + ", std=" + str(np.std(preds_nonvap_m[1])))
    print("\t\tTotal: mean=" + str(np.mean(preds_total_nonvap_m)) + ", std=" + str(np.std(preds_total_nonvap_m)))

    print("Female")
    if test_size != 0:
        print("\tCounts: Train=" + str(len(y_train_f)) + ", Test=" + str(len(y_test_f)))
    else:
        print("\tCounts: Train=" + str(len(y_train_f)))

    print("\tActivations Vaporfly:")
    print("\t\tNormal: mean=" + str(np.mean(preds_vap_f[0])) + ", std=" + str(np.std(preds_vap_f[0])))
    print("\t\tShoe: mean=" + str(np.mean(preds_vap_f[1])) + ", std=" + str(np.std(preds_vap_f[1])))
    print("\t\tTotal: mean=" + str(np.mean(preds_total_vap_f)) + ", std=" + str(np.std(preds_total_vap_f)))
    print("\tActivations Non Vaporfly:")
    print("\t\tNormal: mean=" + str(np.mean(preds_nonvap_f[0])) + ", std=" + str(np.std(preds_nonvap_f[0])))
    print("\t\tShoe: mean=" + str(np.mean(preds_nonvap_f[1])) + ", std=" + str(np.std(preds_nonvap_f[1])))
    print("\t\tTotal: mean=" + str(np.mean(preds_total_nonvap_f)) + ", std=" + str(np.std(preds_total_nonvap_f)))


    print("Actual times vaporfly:")
    print("\tTotal: mean=" + str(np.mean(y_train_vap)) + ", std=" + str(np.std(y_train_vap)))
    print("\tMale: mean=" + str(np.mean(vap_y_m)) + ", std=" + str(np.std(vap_y_m)))
    print("\tFemale: mean=" + str(np.mean(vap_y_f)) + ", std=" + str(np.std(vap_y_f)))

    print("Actual times nonvaporfly:")
    print("\tTotal: mean=" + str(np.mean(y_train_nonvap)) + ", std=" + str(np.std(y_train_nonvap)))
    print("\tMale: mean=" + str(np.mean(nonvap_y_m)) + ", std=" + str(np.std(nonvap_y_m)))
    print("\tFemale: mean=" + str(np.mean(nonvap_y_f)) + ", std=" + str(np.std(nonvap_y_f)))
    






if __name__ == '__main__':
    compare_models(test_size=0.0, shoe_effects=['race_'])