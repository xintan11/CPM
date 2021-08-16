# This file saves the functions used for the DCDR model

import pandas as pd
import numpy as np
import datetime
from matplotlib import pyplot as plt

# ------------------------------------------------------
# 0. Cut generator and combiner: copied from DCDR
# ------------------------------------------------------



def cut_generator(ncut, minimum, maximum, seed=1234, random=True, 
                  empirical_data=None, dist='uniform'):
    """
    Return equal-spaced or quantile-based bins 

    random: boolean
       Whether to get exact bins or add random variation to the bins by picking cuts randomly from a uniform 
       distribution or picking the quantiles to get from the data from a uniform distribution

    empirical_data: array-like
    dist: str, ["uniform", empirical]
        Uniform returns equal-distance bins, empirical returns quantile-based bins.
        Empirical requires empirical_data to be an array with the y data
    """
    if random:
        np.random.seed(seed)
        if dist=='empirical' and (empirical_data is not None):
            qt_cut = np.random.uniform(0, 100, size=ncut)
            cut_points = np.percentile(empirical_data, qt_cut)
        elif dist=='uniform':
            cut_points = np.random.uniform(minimum, maximum, ncut)
    else:
        if dist=='empirical' and (empirical_data is not None):
            qt_cut = np.linspace(0, 100, num=ncut)
            cut_points = np.percentile(empirical_data, qt_cut)
        elif dist=='uniform':
            cut_points = np.linspace(minimum, maximum, num=ncut) 

    cut_points = np.sort(cut_points)

    return cut_points

def cut_combiner(cut_points, train_y):
    """Remove duplicated cutpoints and cutpoints that lead to empty bins

    Parameters
    ----------
    cut_points : list of cutpoints
    train_y : array-like

    Returns
    -------
    list of cutpoints
    """
    # TODO: Seems to be failing for quantiles and returns duplicates
    # REASON: the cut_points was in float format, forcing int removes the duplicates
    cut_points = np.unique(cut_points.astype(int))    
    
    idx = np.digitize(train_y, cut_points)
    right_idx = np.unique(idx)
    left_idx = right_idx-1
    all_valid_idx = np.union1d(left_idx, right_idx)
    all_valid_idx = all_valid_idx[(all_valid_idx>=0) & (all_valid_idx<len(cut_points))]

    return cut_points[all_valid_idx]

# ------------------------------------------------------
# 1. Feature Engineering
# ------------------------------------------------------



def feature_engineering(df):

    # (1) Ad Width x Height
    df['Ad Slot Area'] = df['Ad Slot Width'] * df['Ad Slot Height']

    def convert_timestamp(x: np.int64):
        return datetime.datetime.strptime(str(x), '%Y%m%d%H%M%S%f').timestamp()
        
    # (2) Convert time to seconds, then calculate sin / cos per minute
    # The freaquency of minute is found by rfft
    # c.f. https://www.tensorflow.org/tutorials/structured_data/time_series

    timestamp_s = df['Timestamp'].apply(lambda x:convert_timestamp(x)) 
    df['Minute sin'] = np.sin(timestamp_s * (2 * np.pi / 60)) # 60 seconds per minute
    df['Minute cos'] = np.cos(timestamp_s * (2 * np.pi / 60))


    # (3) All IP addresses in the file has the form '114.100.37.*', convert to int

    def IP_to_num(ip: str):
        sub_strings = ip.split('.')
        sub_numbers = [int(x) for x in sub_strings[:3]]
        return sub_numbers[0] * 256*256 + sub_numbers[1] * 256 + sub_numbers[2]

    df['IP_numeric'] = df['IP'].map(IP_to_num)    

    # (4) User Agent: AppleWebKit: 1, Android: 2, MSIE: 4, iPad or iPhone: 8

    df['UserAgent Types'] = (df['UserAgent'].str.contains('iPad') | df['UserAgent'].str.contains('iPhone')) * 8
    for idx, s in enumerate(['AppleWebKit', 'Android', 'MSIE']): 
        df['UserAgent Types'] += 2**idx * df['UserAgent'].str.contains(s)
        
    df['UserAgent Types'] = df['UserAgent Types'].fillna(0)

    return df

# ------------------------------------------------------
# 2. Tensorflow Layers
# ------------------------------------------------------

# 2.1 df to dataset
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

# Labels in one_hot encoding from 'Paying Price' (Market Price)
def df_to_dataset(dataframe, cuts, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    dataframe = dataframe.drop(columns=['Log Type', 'Anonymous URL', 'Bidding Price'])
    labels = dataframe.pop('Paying Price')
    labels = np.digitize(labels, cuts)
    labels = tf.one_hot(labels, depth=len(cuts)+1)

    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds

# Labels as bin number from 'Paying Price' (Market Price), without the one_hot step
# testing purpose
def df_to_dataset_without_onehot(dataframe, cuts, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    dataframe = dataframe.drop(columns=['Log Type', 'Anonymous URL', 'Bidding Price'])
    labels = dataframe.pop('Paying Price')
    labels = np.digitize(labels, cuts)
    
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds

# Labels with 2 columns: ['Paying Price', 'Bidding Price'] (Market, Bid), both as bin numbers without one_hot
# pay attention to the order of the two prices, matching that in the custom loss function
def df_to_dataset_censored(dataframe, cuts, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe[['Paying Price', 'Bidding Price']].copy()
    dataframe = dataframe.drop(columns=['Log Type', 'Anonymous URL', 'Paying Price', 'Bidding Price'])
    
    labels = np.digitize(labels, cuts)
    
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds


# 2.2 Preprocessing layers
# https://www.tensorflow.org/tutorials/structured_data/preprocessing_layers

def get_normalization_layer(name, dataset):
    # Create a Normalization layer for our feature.
    normalizer = preprocessing.Normalization(axis=None)

    # Prepare a Dataset that only yields our feature.
    feature_ds = dataset.map(lambda x, y: x[name])

    # Learn the statistics of the data.
    normalizer.adapt(feature_ds)

    return normalizer

def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
    # Create a StringLookup layer which will turn strings into integer indices
    if dtype == 'string':
        index = preprocessing.StringLookup(max_tokens=max_tokens)
    else:
        index = preprocessing.IntegerLookup(max_tokens=max_tokens)

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])

    # Learn the set of possible values and assign them a fixed integer index.
    index.adapt(feature_ds)

    # Create a Discretization for our integer indices.
    encoder = preprocessing.CategoryEncoding(num_tokens=index.vocabulary_size())

    # Apply one-hot encoding to our indices. The lambda function captures the
    # layer so we can use them, or include them in the functional model later.
    return lambda feature: encoder(index(feature))

    # Try embedding? 
    # feature_embedding = tf.keras.layers.Embedding(
    #     input_dim = index.vocabulary_size(),
    #     output_dim = 8
    # )
    
    # flat = tf.keras.layers.Flatten()
    
    # return lambda feature: flat(feature_embedding(index(feature)))

# https://www.tensorflow.org/recommenders/examples/featurization#using_feature_hashing
def get_hashing_embedding_layer(name, dataset, dtype, max_tokens, output_dim=32):
    
    if dtype == 'string':
        index = preprocessing.StringLookup(max_tokens=max_tokens)
    else:
        index = preprocessing.IntegerLookup(max_tokens=max_tokens)
    
    feature_ds = dataset.map(lambda x, y: x[name])
    index.adapt(feature_ds)
        
    # num_hashing_bins = 100_000
    num_hashing_bins = index.vocabulary_size()
    if max_tokens is not None:
        num_hashing_bins = min(num_hashing_bins, max_tokens)
        
    feature_hashing = preprocessing.Hashing(num_bins=num_hashing_bins)
    
    feature_embedding = tf.keras.layers.Embedding(
        input_dim = num_hashing_bins,
        output_dim = output_dim
    )
    
    flat = tf.keras.layers.Flatten()
    
    # This also works, it just adds another Input layer
    # return tf.keras.Sequential([feature_hashing, feature_embedding, tf.keras.layers.Flatten()])
    
    return lambda feature: flat(feature_embedding(feature_hashing(index(feature))))


# ------------------------------------------------------
# 3. Loss Functions
# ------------------------------------------------------

# 3.1 Loss functions from DCDR
# Loss function definitions
def binary_loss(y_true, y_pred):
    # Johannes: I think this is just a manual implementation of the categorical cross entropy in TF?
    loss = 0
    # Clip extreme predictions to ensure numerical stability of log
    clipped_y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    # Log loss
    loss += -tf.reduce_mean(tf.math.log(clipped_y_pred) * y_true)
    loss += -tf.reduce_mean(tf.math.log(1 - clipped_y_pred) * (1 - y_true))
    return loss

def crps_loss(y_true, y_pred):
    """
    Continuous ranked probability score
    Mean squared error between cumulative probability of bin y and indicator if y >= y_true
    See Li et al (2019) Eq. 9
    """
    loss = 0
    clipped_y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)    
    loss += tf.reduce_mean(tf.square(1-clipped_y_pred) * y_true)
    loss += tf.reduce_mean(tf.square(clipped_y_pred) * (1 - y_true))
    return loss

# 3.2 Loss functions for y_true in one_hot encoding
def my_loss(lag = 5):
    
    def my_binary_loss(y_true, y_pred):

        cdf_true = y_true

        # Shift y_true rightwards to allow for slight overbid
        # [0 0 1 0 0] -> [0 0 1 .66 .33] for lag = 2
        paddings = [[0, 0], [0, lag]]
        for i in range(lag):
            cdf_true += (lag-i) / (lag+1) * tf.roll(tf.pad(y_true, paddings), shift=i, axis=1)[:, :-lag]

        lost_bins = (1 - cdf_true) * tf.math.log( tf.clip_by_value(1 - y_pred, 1e-20, 1) )
        win_bins  = cdf_true * tf.math.log( tf.clip_by_value( y_pred, 1e-20, 1) )

        return -tf.reduce_sum( lost_bins + win_bins )
    
    return my_binary_loss

def cdf_loss(y_true, y_pred):
    
    cdf_true = tf.math.cumsum(y_true) # tf.clip_by_value(tf.math.cumsum(y_true), 1e-10, 1 - 1e-10)
    cdf_pred = tf.math.cumsum(y_pred)
    
    loss = 0
    loss += -tf.reduce_mean(tf.math.log( tf.clip_by_value(cdf_pred, 1e-20, 1 )) * cdf_true)
    loss += -tf.reduce_mean(tf.math.log( tf.clip_by_value(1 - cdf_pred, 1e-20, 1 )) * (1 - cdf_true))
    
    return loss

# 3.3 Censored loss function with y_true = ['Paying Price', 'Bidding Price']
# Similar idea to Ren et al. 2019's Deep Landscape Forecasting
# The final loss is the weighted sum of two parts: 
# (1) Binary Cross Entropy from Market Price (Paying Price, 2nd Price), if censored = True, then losing bids are cencored
# (2) Cross Entropy from Bidding Price.cumsum (~ CDF), 
#     where losing bids have losses from all bins incl. and lower than then bidding price, 
#     and winning bids have losses from bins incl. and above the bidding price. 
# <=>
# Winning CE: y_true.cumsum * log ( y_pred.cumsum )
# Losing CE: ( 1 - y_true.cumsum + y_true ) * log ( 1 - y_pred.cumsum )

def my_censored_binary_loss(ce_loss_weight = 0.25, censored = True):

    def my_binary_loss(y_true, y_pred):
        market_price = y_true[:, 0]
        bid_price = y_true[:, 1]
        winning = tf.cast(bid_price > market_price, tf.float32) # tf calculation requires float * float (not float * int) 

        num_bins = tf.cast(y_pred.shape[-1], tf.int32)
        market_one_hot = tf.one_hot(market_price, depth=num_bins)
        bid_one_hot = tf.one_hot(bid_price, depth=num_bins)
        
        clipped_y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7) # why 1e-7? 1e-10 is too small -> nan (also in Ren et al.)
        cumsum_y_pred  = tf.clip_by_value(tf.math.cumsum(y_pred, axis=-1), 1e-7, 1 - 1e-7)

        if censored:
            # Losing bids are censored: don't know the true market price -> don't count their NLL loss
            binary_loss =  -tf.reduce_sum(tf.math.log(clipped_y_pred) * market_one_hot, 
                                          axis=-1) * winning
            binary_loss += -tf.reduce_sum(tf.math.log(1 - clipped_y_pred) * (1 - market_one_hot), 
                                          axis=-1) * winning 
            binary_loss = tf.reduce_sum(binary_loss)
        else:
            binary_loss =  -tf.reduce_sum(tf.math.log(clipped_y_pred) * market_one_hot)
            binary_loss += -tf.reduce_sum(tf.math.log(1 - clipped_y_pred) * (1 - market_one_hot))
        
        winning_CE = tf.reduce_sum(
            tf.math.cumsum(bid_one_hot, axis=-1) * -tf.math.log(cumsum_y_pred), # maybe using market_one_hot is a better idea
            axis = -1
        )

        # (1 - bid.cumsum + bid) * log (1 - pred) + market.cumsum * log (pred), 
        # since market in this dataset is the highest bid for the losing records.
        # Essentially, this push down the CDF below bid and push up the CDF above market = highest bid
        losing_CE = tf.reduce_sum(
            (1 - tf.math.cumsum(bid_one_hot, axis=-1) + bid_one_hot) * -tf.math.log(1 - cumsum_y_pred)\
            + tf.math.cumsum(market_one_hot, axis=-1) * -tf.math.log(cumsum_y_pred), 
            axis = -1 
        )
        cross_entropy = tf.reduce_sum(
            winning * winning_CE + (1-winning) * losing_CE
        )
        
        combined_loss = (1.0-ce_loss_weight) * binary_loss + ce_loss_weight * cross_entropy
        return combined_loss
    return my_binary_loss




# ------------------------------------------------------
# 4. Plotting and Accuracy Checks
# ------------------------------------------------------

# 4.1 Plot bin predictions, predicted CDF, market and bid prices
# 
# Sample call:
# plot_predictions(predictions=predictions, targets=test['Paying Price'], _
#                  bids=test['Bidding Price'], example=123, save=False)

def plot_predictions(predictions, targets, bids, bins, example, save=False):
    
    fig, ax1 = plt.subplots(figsize=(15, 5))
    ax2 = ax1.twinx()
    ax1.plot(bins[1:], predictions[example].cumsum(), color='blue') # plot(x, y)
    ax2.plot(bins[1:], predictions[example], color='green', alpha=0.5)
    
    ax1.vlines(targets.values[example], 0, 1, color='red')
    ax1.vlines(bids.values[example], 0, 1, color='orange')
    
    ax1.legend(['Winning Prob.', 'Paying Price', 'Bidding Price'], loc='upper left')
    ax2.legend(['Predictions'], loc='lower right')
    
    plt.title('Prediction CDF for Example #{}'.format(example))
    plt.grid()
    ax1.set_xlabel('Price')
    ax1.set_ylabel('Winning Prob.')
    ax2.set_ylabel('Predicted Bin Prob. Density')
    
    if(save):
        plt.savefig('z_predictions_'+str(example)+'.png')
        plt.close()



# 4.2 Calculate the % of winning bids within the test sample
# 
# Assume that if market price in bin #2, and predicted bin = 2, then win
# (implies bidding the right edge of the bin)
# 
# Sample call:
# calc_prediction_accuracy(predictions, test['Paying Price'], cuts, 0.5)

def calc_prediction_accuracy(predictions, targets, cuts, percentile_to_bid):
    assert predictions.shape[0] == targets.shape[0]
    
    prediction_cdf = predictions.cumsum(axis=1)  # predictions: numpy array
    
    # We want the bin left of the true bin, hence -1
    price_bins = np.digitize(targets, cuts) - 1  
    
    true_bins  = tf.one_hot(price_bins, depth=predictions.shape[1]).numpy() # only the correct bin = 1, others = 0
    
    # true_bins * cdf = the predicted prob. of winning at a bin lower than the market price
    # if predicted prob. < %_to_bid, then the bid must be in a bin >= of the market price bin
    # Note < instead of <= compared to previous version
    
    return ((true_bins * prediction_cdf).sum(axis=1) < percentile_to_bid).sum() / predictions.shape[0]
