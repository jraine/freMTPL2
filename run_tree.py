import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

from utils import load_merged_dataframe, cat_to_onehot, mean_std_floats, inverse_transform_floats

from plot import plot_overlay

np.random.seed(42)

def main():
    #Load data from datapath
    datapath = pathlib.Path('./data/')
    freq_datapath = datapath / 'freMTPL2freq.arff'
    sev_datapath = datapath / 'freMTPL2sev.arff'
    df = load_merged_dataframe(freq_datapath,sev_datapath)
    
    #Add potential target features
    df['YearCost'] = df.ClaimAmount/df.Exposure
    df['ClaimYear'] = (df.ClaimNb/df.Exposure).fillna(0)
    df['ClaimCost'] = (df.ClaimAmount/df.ClaimNb).fillna(0)

    # Clean data
    df = df[df['YearCost'] < 1e5]

    # Get index for splitting into train, test, validation (0.6,0.2,0.2)
    idx = np.arange(0,len(df),1)
    np.random.shuffle(idx)

    # Choose input features to method
    # VehAge, DrivAge, BonusMalus, Density: floats
    floatvars = ['VehAge', 'DrivAge', 'BonusMalus', 'Density']
    # Area, VehBrand, VehGas, VehPower, Region: categorical
    catvars =  ['Area', 'VehBrand', 'VehGas', 'VehPower', 'Region']
    
    # Get training set, use these for float conversion
    x_float_train, float_trafo = mean_std_floats(df.iloc[idx[:int(0.6*len(df))]],
                                                 floatvars)

    x_float_val,_ = mean_std_floats(df.iloc[idx[int(0.6*len(df)):int(0.8*len(df))]],
                                  floatvars,
                                  float_trafo)

    x_float_test,_ = mean_std_floats(df.iloc[idx[int(0.8*len(df)):]],
                                  floatvars,
                                  float_trafo)

    # Get float vars, convert to onehot
    x_cat = cat_to_onehot(df,catvars)
    x_cat_train = x_cat[idx[:int(0.6*len(df))]]
    x_cat_val = x_cat[idx[int(0.6*len(df)):int(0.8*len(df))]]
    x_cat_test = x_cat[idx[int(0.8*len(df)):]]
    
    # Get target, learn scaling from train
    y_train, target_trafo = mean_std_floats(df.iloc[idx[:int(0.6*len(df))]],
                                            ['YearCost'])

                              
    y_val_t,_ = mean_std_floats(df.iloc[idx[int(0.6*len(df)):int(0.8*len(df))]],
                              ['YearCost'],
                              target_trafo)

    y_test_t,_ = mean_std_floats(df.iloc[idx[int(0.8*len(df)):]],
                                   ['YearCost'],
                                   target_trafo)

    # Plot the float distributions and target
    plot_overlay(x_float_train, floatvars, './results/', 'inputs_scaled')
    plot_overlay(y_train, ['YearCost'], './results/', 'target_scaled')
    


    # Hyperparameters for boosted regression tree
    # Preferably specified in config file or passed as cmd args
    # e.g. Use of argparse or hydra+omegaconf
    criterion = 'squared_error'
    learning_rate = 0.1
    n_estimators = 100
    subsample = 1.0
    max_depth = 4
    min_samples_split = 2
    max_features = None

    # Would also specify this in a config file
    # Time constraints limit flexibility of code
    # But certainly intersting to compare the sets of variables
    # Would also add a feature passing argument for config to specify which to run on
    # Would also want to compare this to using a feedforward MLP for regression
    # For MLP would ideally have two tasks - predict claim number per year, and amount per claim; multiply the claim number by amount to get year cost

    do_all = True
    do_float = False
    do_cat = False

    # Get validation set dataframe
    val_df = df.iloc[idx[int(0.6*len(df)):int(0.8*len(df))]]

    if do_all:
        # Define Boosted regressor, train on all inputs
        regressor_all = GradientBoostingRegressor(criterion=criterion,
                                            learning_rate=learning_rate,
                                            n_estimators=n_estimators,
                                            subsample=subsample,
                                            max_depth=max_depth,
                                            min_samples_split=min_samples_split,
                                            max_features=max_features,
                                            random_state=0)

        regressor_all.fit(np.concatenate([x_float_train,x_cat_train],axis=1),y_train.ravel())
        val_pred = target_trafo.inverse_transform(regressor_all.predict(np.concatenate([x_float_val,x_cat_val],axis=1))).ravel()
        val_df['regress_all'] = val_pred
    
    if do_float:
        # Define Boosted regressor, train on float inputs
        regressor_float = GradientBoostingRegressor(criterion=criterion,
                                            learning_rate=learning_rate,
                                            n_estimators=n_estimators,
                                            subsample=subsample,
                                            max_depth=max_depth,
                                            min_samples_split=min_samples_split,
                                            max_features=max_features,
                                            random_state=0)

        regressor_float.fit(x_float_train,y_train.ravel())
        val_pred = target_trafo.inverse_transform(regressor_all.predict(x_float_val)).ravel()
        val_df['regress_float'] = val_pred

    if do_cat:
        # Define Boosted regressor, train on cat inputs
        regressor_cat = GradientBoostingRegressor(criterion=criterion,
                                            learning_rate=learning_rate,
                                            n_estimators=n_estimators,
                                            subsample=subsample,
                                            max_depth=max_depth,
                                            min_samples_split=min_samples_split,
                                            max_features=max_features,
                                            random_state=0)

        regressor_cat.fit(x_cat_train,y_train.ravel())
        val_pred = target_trafo.inverse_transform(regressor_all.predict(x_cat_val)).ravel()
        val_df['regress_cat'] = val_pred

    # Save output scores for further study and plotting
    # Can use this for metrics
    val_df.to_hdf('./results/output_df.h5','val')

    # Measures of interest:
    ## Regression accuracy, look at mean square error, mean absolute error
    ## Look at assumed cost loss, how much under/overestimated
    

if __name__ == "__main__":
    main()