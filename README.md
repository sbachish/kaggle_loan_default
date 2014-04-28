Two stage approach, first stage - classification, optimizing F1-score (AUC gives slightly worse result), 
need something robust, so logit is a good choice.
Second stage - regression, train gradient boosting regression model with LAD loss function on non-zero values 
from previous stage (evaluatiom metric for this competition is MAE).
