# CombSleepNet
Hyeong-Jin Kim, Minji Lee, Dong-Ok Won, and Seong-Whan Lee, "CombSleepNet: Combination of Spectral–Temporal CNN and Bi-LSTM with Dense Connectivity for Automatic Sleep Stage Scoring," submit to IEEE Journal of Biomedical and Health Informatics.

<img src="/img/fig1.png" width="100%" height="100%" title="CombSleepNet" alt="Overview of the CombSleepNet architecture"></img>

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
   + Run the script to train CombSleepNet 
   
   ```python train.py --data_dir "./example_data/" --out_dir "./parameter/" --seq_len 10 --cnn_lr 1e-5 --lstm_lr 1e-3 --cnn_epoch 30 --lstm_epoch 15 --cv 1```
   
   + Run the script to test CombSleepNet
   
   ```python test.py --data_dir "./example_data/" --parameter_dir "./parameter/" --out_dir "./result/" --seq_len 10 --cnn_lr 1e-5 --lstm_lr 1e-3 --cv 1```
