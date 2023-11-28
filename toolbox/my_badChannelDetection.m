function [badChannel,badChannelRMS,badChannelCORR] = ...
    my_badChannelDetection(emg,win_len,step_len,fs_emg)

% 9.13: add rms
rms_tmp = get_rms(emg,win_len,step_len,fs_emg);
rms = mean(rms_tmp,1);
tempRMSThr1 = mean(rms) + 2*std(rms);
tempRMSThr2 = mean(rms) - 2*std(rms);
badChannelRMS = find(rms>tempRMSThr1 | rms<tempRMSThr2);

corr_tmp = mean( corrcoef(emg));
% tempCORRThr1 = mean(corr_tmp) +3*std(corr_tmp);
tempCORRThr2 = mean(corr_tmp) -3*std(corr_tmp);
badChannelCORR = find(corr_tmp<tempCORRThr2);

badChannel = unique([badChannelRMS,badChannelCORR]);

end