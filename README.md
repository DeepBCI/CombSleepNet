# CombSleepNet
Hyeong-Jin Kim, Minji Lee, Dong-Ok Won, and Seong-Whan Lee, "CombSleepNet: Combination of Spectral–Temporal CNN and Bi-LSTM with Dense Connectivity for Automatic Sleep Stage Scoring," submit to IEEE Journal of Biomedical and Health Informatics.

<img src="/img/fig1.png" width="100%" height="100%"></img>

## How to run
1. Download the Sleep-EDF database
   + Sleep-EDF database is available [here][sleep-edf].
   
[sleep-edf]: https://physionet.org/content/sleep-edfx/1.0.0/
   
2. Data pre-processing
   + Change directory to ```./CombSleepNet/pre-processing```
   + Unzip ```eeflab.zip```
   + Run ```preprocessing.m```
   
3. Training and testing the CombSleepNet
   + Change directory to ```./CombSleepNet```
   + Refer to ```example.txt``` to train and test CombSleepNet.
   
## Environment:
+ Matlab R2019b
+ Python3
+ Pytorch v1.3.1
+ numpy v1.17.4
+ scipy v1.3.3
+ scikit-learn v0.22

## Result:
Hypnogram and posterior probability distribution with CombSleepNet for one subject of Sleep-EDF database
<img src="/img/fig3.png" width="100%" height="100%"></img>
