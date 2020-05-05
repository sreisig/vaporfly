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

from util import *

def compare_models(test_size=0.2, shoe_effects=['race_', 'gender']):

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
    an, activation_network = get_additive_model(cols, shoe_effects=shoe_effects)
    # construct input vectors
    x_train_an, new_cols = get_additive_inputs(x_train, cols, shoe_effects=shoe_effects, get_cols=True)
    if test_size != 0:
        x_test_an = get_additive_inputs(x_test, cols, shoe_effects=shoe_effects)

    an.compile(optimizer='adam', loss='mean_squared_error')

    if test_size != 0:
        hist = an.fit(x=x_train_an, y=y_train, batch_size = 32, epochs=200, validation_data = (x_test_an, y_test), callbacks=[], verbose=0)
    else:
        hist = an.fit(x=x_train_an, y=y_train, batch_size = 32, epochs=1000, callbacks=[], verbose=0)

    an_train_r2 = r2_score(y_train, an.predict(x_train_an))
    if test_size != 0:
        an_test_r2 = r2_score(y_test, an.predict(x_test_an))


    # get activations for vaporfly and non vaporfly
    # not really sure if i should be doing this on train data, test data, or both
    vap_x, vap_y = get_vaporfly_inputs(x_train_an, y_train)
    nonvap_x, nonvap_y = get_vaporfly_inputs(x_train_an, y_train, inverse=True)
    # print (vap_x)
    # print (nonvap_x)
    # print (np.mean(vap_y))
    # print (np.mean(nonvap_y))

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