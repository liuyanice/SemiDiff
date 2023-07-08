GPU=1
seed_split=0  # data split seed
Mode=train  # training (train) or testing (test) mode
Dataset=DRIVE # dataset to use [option: STARE, CHASE DB, EM and DRIVE]

Epoch=1000  # number of training epochs
lr=1e-3 # init learning rate
bs=8  # batchsize
SaveRslt=1  # save experiment results flag
labelpercent=0.01 # labelled ratio (0 to 1)
ssl=MeanTeacher # semi-supervised learner
EmaAlpha=0.999  # EMA hyperparameter
Gamma=5 # weight applied to unsupervised loss
RampupType=Exp  # unsupervised loss weight rampup method

## SemiDif
RampupEpoch=500
net=ResUnet
loss=Dice+ConsistMSE_loss # supervised loss
python main.py --GPU $GPU --net $net --Mode $Mode --Epoch $Epoch --LearningRate $lr --batchsize $bs --SaveRslt $SaveRslt \
 --labelpercent $labelpercent --seed_split $seed_split --loss $loss --ssl $ssl --Alpha $EmaAlpha --Gamma $Gamma \
 -ta $Dataset -vf 2 --MaxTrIter 5 --RampupEpoch $RampupEpoch --RampupType $RampupType


## SemiCurv
#RampupEpoch=500
#net=ResUnet_SinusoidLocation
#loss=Dice+CosineContrastive_loss
#python main.py --GPU $GPU --net $net --Mode $Mode --Epoch $Epoch --LearningRate $lr --batchsize $bs --SaveRslt $SaveRslt \
 --labelpercent $labelpercent --seed_split $seed_split --loss $loss --ssl $ssl --Alpha $EmaAlpha --Gamma $Gamma \
 -ta $Dataset --MaxTrIter 5 --RampupEpoch $RampupEpoch --RampupType $RampupType
 
 
## SemiDif command line
#python main.py --GPU 0 --net ResUnet --Mode train --Epoch 1000 --LearningRate 1e-3 --batchsize 8 --SaveRslt 1 --labelpercent 0.01 --seed_split 0 --loss Dice+ConsistMSE_loss --ssl MeanTeacher --Alpha 0.999 --Gamma 5 -ta DRIVE -vf 2 --MaxTrIter 5 --RampupEpoch 500 --RampupType Exp


