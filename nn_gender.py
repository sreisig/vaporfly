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

from util import *


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