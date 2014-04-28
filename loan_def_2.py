import numpy as np
import pandas as pd

import cPickle as pickle

from sklearn import linear_model, svm, ensemble, metrics
from sklearn.cross_validation import train_test_split


# if prob > thresh => default (i.e. 1)
def mae_predict(trainX,trainY,model, thresh=0.2):

    # good features from cv
    with open('golden_mae_test.dat', 'rb') as infile:
        testX = pickle.load(infile)
    print trainX.shape, trainY.shape, testX.shape

    pred_f1 = np.array(pd.read_table('f1.csv', sep=','))
    print pred_f1.shape

    # I don't remember why, I guess some features were removed from competition
    testX = np.delete(testX,[15,16],1) 

    prob_f1 = pred_f1[:,1]
    prob_f1[prob_f1>=thresh]=1
    prob_f1[prob_f1<thresh]=0
    print 'def sum:', np.sum(prob_f1)
    test_def = testX[prob_f1>=1]
    print test_def.shape


    model.random_state = 34372
    model.max_depth = 8

    print 'Training...'
    model.fit(trainX, trainY)
    print 'Predicting...'
    preds = model.predict(test_def)

    pr = preds
    pr = preds.reshape(-1,1) 
    pr = np.array(pr, dtype='float') 

    # training and averaging (20+1 models)
    for k in range(20):
        print k+1
        model.random_state += k*1780
        model.fit(trainX, trainY)
        preds = model.predict(test_def)
        pr = np.hstack([ pr, preds.reshape(-1,1)  ])

    pr = np.mean(pr,1)
    pr = np.floor(pr)
    pr = np.array(pr, dtype='int' )
    preds = pr


    prob_f1[prob_f1>=1] = preds.reshape(-1,1)
    prob_f1 = np.floor(prob_f1)
    pred_f1[:,1] = prob_f1
    print pred_f1.shape

    print 'Saving submission...'
    head = 'id,loss'
    col = '%i,'+'%i'
    np.savetxt('_subm17_2.csv', pred_f1, delimiter=",", header=head, comments='',  fmt=col )
#-----------------------------------------------------------------------------------------------

def cv_loop(X, y, model, N, SEED=25):
    MAEs = 0
    for i in range(N):
        
        X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.31, random_state = i*SEED)
        model.fit(X_train, y_train)

        #print np.argmax(model.feature_importances_)

        preds = model.predict(X_cv)
        mae = metrics.mean_absolute_error(y_cv,preds)
        #print "MAE (fold %d/%d): %f" % (i + 1, N, mae)
        MAEs += mae
    return MAEs/N
#-----------------------------------------------------------------------------------------------

def main():

    #data_summary()
    print 'Loading data...'
    data_ = pd.read_table('train.csv', sep=',')     
    # fill NA             
    for i in data_.columns:
        data_[i] = data_[i].fillna(data_[i].median(0))
    data_ = np.array(data_)
    print data_.shape

    # only defaults (i.e. non-zeros targets)
    data_ = data_[data_[:,-1]>0]
    print data_.shape

    trainY = data_[:,-1]
    trainX = data_[:,0:-1]
    print np.isnan(trainX).any()

    # good features for regression
    with open('golden_mae.dat', 'rb') as infile:
        golden_mae = pickle.load(infile)


    # 11752
    model = ensemble.GradientBoostingRegressor(loss='lad', learning_rate=0.1, n_estimators=11,#47
                                               subsample=1.0, min_samples_split=2, min_samples_leaf=1, 
                                               max_depth=1, max_features=1, verbose=0,random_state=None) 
    #model = ensemble.RandomForestRegressor(n_estimators=90, criterion='mse', max_depth=5, min_samples_split=2, 
    #                                       min_samples_leaf=1, max_features=75)

    model.max_depth = 5#5 #4
    model.max_features = 10#10 
    model.n_estimators = 77 #31      #81   cv11

    print model

    # Greedy search, forward stepwise feature selection
    if True:               # False True
        min_mae = 9

        for i in range(trainX.shape[1]):
            if (i+1)%10==0:    print i
           
            X_tr = trainX[:,i].reshape(-1,1)
            X_tr = np.hstack([ golden_mae, X_tr])
                                        
            mae = cv_loop(X_tr, trainY, model,N=7)
            if min_mae > mae:
                min_mae = mae
                new_hero = i
                print i, min_mae

#---------------------------------------------
    # Also, saved good features manually, sorry for this
    # something like this
    if True:     
        golden_mae = np.hstack([ golden_mae, trainX[:,new_hero].reshape(-1,1)  ])   
        with open('golden_mae.dat', 'wb') as outfile:
            pickle.dump(golden_mae, outfile, pickle.HIGHEST_PROTOCOL)
        print 'golden shape:', golden_mae.shape

    # the same for test ...
#---------------------------------------------
 
    # creat a submission
    mae_predict(golden_mae,trainY,model)

    return

#-----------------------------------------------------------------------------------------------
main()
