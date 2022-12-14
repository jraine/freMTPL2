import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

from utils import load_merged_dataframe, cat_to_onehot, mean_std_floats, inverse_transform_floats

from plot import plot_overlay
from network import get_dense

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
    
    # Get target
    y_train = df['ClaimYear'].iloc[idx[:int(0.6*len(df))]]
    y_val = df['ClaimYear'].iloc[idx[int(0.6*len(df)):int(0.8*len(df))]]
    y_test = df['ClaimYear'].iloc[idx[int(0.8*len(df)):]]
                                            

    # Aim here is to predict the number of claims per year from inputs
    # Second task is assuming there is a claim, predict cost
    # Can then calculate estimation over year for each IDpol
    # Benefits over simple regression is restricting sensitivity to occurance based on amount
    # And then no longer need to worry about large range of zeros in amount giving zero-bias


    #Get a categorical network, can see from rate/year that most cases are 0, 1, 2, (3), 4
    # rate_net = get_dense(9,5,0)
    ### train network
    ### need to convert claims/year to nearest integer for targets, then make categorical
    ### next question is whether the classes need weighting, set to equal or weight by cost?
    #### If not right would rather not underestimate the heavy contributors to cost!
    #output can be turned back into a float with weighting of output integers
    
    #equally could try and train with single output but due to high rate of zero unlikely to learn the tail range unless weight each event by YearCost, but need to find a way to preserve rate of predictions at zero
    # rate_net = get_dense(9,1,0)
    #softplus on output, have max value of 5 or so
    #poisson loss function for regression training, assuming rate distribution follows poisson/gamma distribution

    #Get regression network for cost of average claim, we know it is peaked with log range and in counts
    #Need a log transform on target values
    #only needs to train on entries with nonzero avg cost per claim
    # cost_net = get_dense(9,1,0)
    
    #Predict values with expectation being the rate_pred turned back into a float and multiply by prediction from claim cost network
    #This gives expectation for clients per year
