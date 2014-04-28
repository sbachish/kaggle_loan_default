require(caTools)
require(quantreg)
require(Hmisc)
require(ROCR)
require(hydroGOF)
require(datamart)


# trainig data
df = read.csv('train.csv')        

# imputing missing data
for (i in names(df)) {
    if (any(is.na(df[i]))) df[i] <- with(df, impute(df[i], mean))
}

# golden features wich were already found
golden_ = read.csv('golden.csv')  

#------------------- CV part [0.9274952] <- 0.9271879 <- 0.92672 <- 0.9263248
seeds = c(77,2014,135,87,45,742,986,114,65)
for (j in names(df)) {

    #if ( strtoi(strtail(j, -1))<=602 ) next 
    
    # good features
    if (j == 'f2')  next
    if (j == 'f41')  next
    if (j == 'f262')  next
    if (j == 'f271')  next
    if (j == 'f316')  next
    if (j == 'f333')  next
    if (j == 'f334')  next
    if (j == 'f527')  next
    if (j == 'f528')  next

    scr = 0

    tt = golden_
    tt[j] = df[j]
    tt = tt[,c( j, 'f2', 'f41', 'f262', 'f271', 'f316', 'f333', 'f334', 'f527', 'f528', 'loss' )]

    tt$f2 = as.factor(tt$f2)

    for (i in seeds){
        set.seed(i)
        tt = tt[sample(nrow(tt)),]

        # cv proportions 70:30
        train = tt[1:(0.7*nrow(tt)),]
        test = tt[-(1:(0.7*nrow(tt))),]

        # binarizing target
        train[,ncol(train)] = ifelse(train[,ncol(train)]>0,1,0)
        test[,ncol(test)] = ifelse(test[,ncol(test)]>0,1,0)

        # training, predicting, evaluating
        model = glm(as.factor(train[,ncol(train)])~.,data=train[,-ncol(train)],family=binomial)
        pr = predict(model, test[,-ncol(test)],type="response")
        pr = ifelse(is.na(pr),mean(!is.na(pr)),pr)
        pred <- prediction( pr, test[,ncol(test)] ) 

        # some annoying things to calculate 'F1'
        perf.f <- performance(pred, "f")
        f1 = perf.f@y.values[[1]]
        f1 = as.numeric(unlist(f1))
        f1 = na.omit(f1)
        #cat('F1:',max(f1),'AUC:',colAUC(pr,test[,ncol(test)]),'\n')
        scr = scr + max(f1)
    }
    #  0.9274952
    if ( (scr/length(seeds)) > 0.9276 ) cat('nice',j, scr/length(seeds),'\n' ) 
    cat(j, scr/length(seeds),'\n' )

}
#---------------------------------------------------------------------------------------

# I saved good features manually, don't include this part, nothing interesting

#---------------------------------------------------------------------------------------
# test data
df_test = read.csv('test.csv')        

# imputing missing data
# I know, it's a data snooping, but kaggle allows to do this
for (i in names(df_test)) {
    if (any(is.na(df_test[i]))) df_test[i] <- with(df_test, impute(df_test[i], mean))
}

golden_test_ = read.csv('golden_test.csv')  

# then traing on full training data and predict probabilities of non-default for test, save them
# the same code as above, but without splitting, nothing interesting
