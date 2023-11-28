# Multi-Souce-CDEM-Demo
A Demo:   A Robust Engineering Framework of Cross-subject Gestrue Classification for Real-time Implementation based on High-density sEMG

## Brief Introduction
We release a demo of our work. The work focuses on gesture classification based on high-density sEMG(HD-sEMG) in cross-subject and real-time scenario. Cross-subject means that we apply transfer learning(method: CDEM) to calibrate model on new users.  Real-time means that we simulate getting data continuosly from EMG signal collecting device.  

See details in article:  
        A Robust Engineering Framework of Cross-subject Gesture Classification for Real-time Implementation based on High-density sEMG  
        
Source code of CDEM:
        https://github.com/yuntaodu/CDEM  
Paper of CDEM:
        Du Y, Chen Y, Cui F, et al. Cross-domain error minimization for unsupervised domain adaptation[C]//Database Systems for Advanced Applications: 26th International Conference, DASFAA 2021, Taipei, Taiwan, April 11–14, 2021, Proceedings, Part II 26. Springer International Publishing, 2021: 429-448.

## Files Description
### main.mlapp
This is the main GUI code file, including calibration part and prediction part. 
#### Calibration
In this part, we load ten samples (one sample per gesture, *10* gestures in all) from a new user. 

We can choose path to save model and name the model. 

**Input**: See details in **'calibration_data'** part.

**Output**: See details in **'model'** part.

#### Prediction
In this part, we load data stream and model (trained in the calibration part) to stimulate real-time scenario and predict gesture labels. We load label stream so that we can show the true label and compare our predict label with the true label. 

**Input**:  

data stream and label stream (see details in **'test_data'** part).

model (trained in the calibration part)
        
**Output**:   

Prediction Label Stream, true label stream, EMG signal (shown on GUI, update per *0.1* second)

p.s. Raw EMG signal includes *256* channels, but showing all of them may spend too many computing resources. We choose *8* channels (*2* channels(the first channel and the last channel) per electrode) instead.

### calibration_data
The data should contain four parts.  

                Xt_motion:    gesture samples in target domain. ( gesture sample number * feature dim number)  
                Xt_rest:      rest samples in target domain. ( rest sample number * feature dim number )  
                Yt_motion:    gesture labels in target domain. ( 1 * gesture sample number )  
                Yt_rest:      rest labels in target domain. ( 1 * rest sample number )  
  
Here, *gesture sample number* is *100* (*10* actions * *10* windows per action).  
Rest *sample number* is *60* (*10* actions * *6* windows per action).   
Feature *dim number* is *1024* (*256* channels * *4* features per channel).  

### model
Model including *6* variables. Variables *'acc_target1'*, *'mdl1'*, *'P_pca1'* are for Model I (the binary classifier, discriminating the samples between rest and gesture conditions). Variables *'acc_target2'*, *'mdl2'*, *'P_pca2'* are for Model II (the ten-class classifier, giving the predicted gesture label of a testing gesture sample).   

The dims of these variables are all *1* * *N*. *N* is the number of models from mulitple source domain (one model per source domain).  

                P_pca:        the PCA projection matrix (to reduce the dim).  ( feature dim number * PCA dim number)  
                acc_target:   weight for corresponding 'mdl'  
                mdl:          include 5 parameters  
                        P:          projection matrix for domain adaption(CDEM)  ( PCA dim number * CDEM dim number)  
                        proj_mean:  center point of project domain (1 * CDEM dim number)  
                        classMeans: center points of classes in projected source domain (num_class * CDEM dim number)   
                        options:    training options in CDEM  
                        num_class:  the class of model (model I:2; model II:10)  
                
Here, *N* is *40* (some index is empty due to the corresponding data cannot be pre-processed). *Feature dim number* is *1024*. *PCA dim number* is *100*. *CDEM dim number* is *20*.  

### picture
*11* pictures are corresponding to *10* gestures and rest.  

### test data
You can run 'data_stream_concat.m' to get 'data_cut.mat' and 'label_cut.mat' so that you can repeat the result in video 'prediction.mp4'. Or you can just use corresponding data file and label file ('data_subjec1_x.mat' and 'label_subject1_x.mat') to simulate real-time gesture classification in only one gesture.  

                data_subject1_x.mat:  raw data stream of one gesture (labeled x). ( (duration time * sampling rate) * raw channel number)  
                label_subject1_x.mat:  raw label stream of one gesture (labeled x). ( (duration time * sampling rate) * 1)  

If you run 'data_stream_concat.m' to get 'data_cut.mat' and 'label_cut.mat':  

                data_cut:  1 * 10 cell including all of the 'data_subject1_x.mat'  
                label_cut:  1 * 10 cell including all of the 'label_subject1_x.mat'  

Here, Sampling rate is *2048*(Hz). Raw channel number is *256*.  
 
### toolbox
The function toolbox.

### train_feature
The preprocessed training data from source domain. One mat file contains four parts.

                Xs_motion:    gesture samples in source domain. ( gesture sample number * feature dim number )
                Xs_rest:      rest samples in source domain. ( rest sample number * feature dim number )
                Ys_motion:    gesture labels in source domain. ( 1 * gesture sample number )
                Ys_rest:      rest labels in source domain. ( rest sample number * 1 )
  
Here, *gesture sample number* and *rest smaple number* is different in each source domain.
*Feature dim number* is *1024* (*256* channels * *4* features per channel).

### video
The result of correctly running 'main.mlapp'.
