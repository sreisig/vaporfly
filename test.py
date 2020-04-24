import pandas as pd
import numpy as np
import os 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from keras.models import Model
from keras.layers import *
from keras import backend as K
from keras.callbacks import EarlyStopping

def load_data(input_folder, one_hot_encode=False, do_name=True):
    cols = ['match_name', 'marathon', 'time_minutes', 'vaporfly'] # not using year column 

    dtypes = {'vaporfly' : np.float64, 'time_minutes' : np.float64} # convert true/false to 1/0 and make sure time is a float

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
            
    if one_hot_encode:
        # one hot encode the predictors match_name and marathon
        one_hot_cols = ['marathon']
        prefix = ['race_']
        if do_name:
            one_hot_cols.append('match_name')
            prefix.append('')
        else:
            df = df[[col for col in df.columns if col != 'match_name']] # get rid of match name
        df = pd.get_dummies(df, columns=one_hot_cols, prefix=prefix, prefix_sep='')

    return df


def get_data(input_folder, one_hot_encode=False, do_name=True, return_cols=False):
    df = load_data(input_folder, one_hot_encode=one_hot_encode, do_name=do_name)

    y_cols = ['time_minutes']
    x_cols = [col for col in df.columns if col not in y_cols]

    x = df[x_cols].values
    y = df[y_cols].values

    if return_cols :
        return x, y, x_cols
    return x, y


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

def get_additive_model(x):
    inlayer_normal = Input(shape=(602, )) #  everything but vaporfly
    inlayer_shoe_effect = Input(shape=(24,)) # for gender plus 21 race courses
    inlayer_shoe = Input(shape=(1,))

    # make normal network
    d = Dropout(0.1)(inlayer_normal)
    temp = Dense(50, activation='relu')(d)
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
    outlayer = Add()([shoe_out, normal_out])

    model = Model(inputs=[inlayer_normal, inlayer_shoe_effect, inlayer_shoe], outputs=outlayer)
    # print(model.summary())

    shoe_effect_model = Model(inputs=[inlayer_shoe_effect, inlayer_shoe], outputs=shoe_out)

    return model, shoe_effect_model



def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def compare_models():
    input_folder = 'data'

    x, y, cols = get_data(input_folder, one_hot_encode=True, do_name=True, return_cols=True)
    # print (cols)

    # really should be doing cross validation
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, shuffle=True)

    linear_model = LinearRegression()
    linear_model.fit(x_train, y_train)

    linear_train_r2 = linear_model.score(x_train, y_train)
    # this fails if do_name is true, because it doesnt get any training points for some of the names
    # need to actually implement random effects model to get better comparison
    linear_test_r2 = linear_model.score(x_test, y_test)

    nn = get_feedforward_model(x_train)
    nn.compile(optimizer='adam', loss='mean_squared_error', metrics=[r2_keras])

    # early stopping to prevent overfitting
    es = EarlyStopping(monitor='val_r2_keras', patience=5, mode='max')
    hist = nn.fit(x=x_train, y=y_train, batch_size = 32, epochs=50, validation_data = (x_test, y_test), callbacks=[es])
    nn_train_r2 = nn.evaluate(x_train, y_train)[1]
    nn_test_r2 = nn.evaluate(x_test, y_test)[1]


    # now do additive network
    an, shoe_model = get_additive_model(x_train)
    # construct input vectors
    x_train_df = pd.DataFrame(x_train, columns=cols)
    train_normal = x_train_df[[col for col in cols if col != 'vaporfly']].values
    train_shoe_effect = x_train_df[[col for col in cols if col=='gender' or 'race_' in col]].values
    train_shoe = x_train_df[[col for col in cols if col == 'vaporfly']].values


    # same for test
    x_test_df = pd.DataFrame(x_test, columns=cols)
    test_normal = x_test_df[[col for col in cols if col != 'vaporfly']].values
    test_shoe_effect = x_test_df[[col for col in cols if col=='gender' or 'race_' in col]].values
    test_shoe = x_test_df[[col for col in cols if col == 'vaporfly']].values


    an.compile(optimizer='adam', loss='mean_squared_error', metrics=[r2_keras])
    es = EarlyStopping(monitor='val_r2_keras', patience=20, mode='max')

    hist = an.fit(x=[train_normal, train_shoe_effect, train_shoe], y=y_train, batch_size = 32, epochs=50, validation_data = ([test_normal, test_shoe_effect, test_shoe], y_test), callbacks=[es])
    an_train_r2 = an.evaluate([train_normal, train_shoe_effect, train_shoe], y_train)[1]
    an_test_r2 = an.evaluate([test_normal, test_shoe_effect, test_shoe], y_test)[1]


    vaporfly_indices = [i for i in range(len(train_shoe)) if train_shoe[i] == 1]
    vaporfly_train = [train_shoe_effect[vaporfly_indices], train_shoe[vaporfly_indices]]

    preds = shoe_model.predict(x=vaporfly_train)
    



    print("Linear Model:")
    print("\tTrain R Squared: " + str(linear_train_r2))
    print("\tTest R Squared: " + str(linear_test_r2))
    print("NN Model:")
    print("\tTrain R Squared: " + str(nn_train_r2))
    print("\tTest R Squared: " + str(nn_test_r2))
    print("Additive Network Model:")
    print("\tTrain R Squared: " + str(an_train_r2))
    print("\tTest R Squared: " + str(an_test_r2))
    print("\tEffect of shoe (train): " + str(np.mean(preds)) + " " + str(np.std(preds)))





if __name__ == '__main__':
    compare_models()