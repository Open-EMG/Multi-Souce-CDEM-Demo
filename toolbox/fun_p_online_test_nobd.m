function [acc_rest_all,acc_gesture_all,acc] = fun_p_online_test_nobd(path_head,path_toolbox,path_data,subjectID,...
    sessionID,subject_test_ID,session_test,index,thresh_classifier2)

addpath(genpath(path_toolbox));
addpath(genpath(path_head));

subject_test = subjectID{subject_test_ID};

cd_mode = 'y';  % 'y' or 'n'

% num_choose = 4;

dim = 100;
zc_ssc_thresh = 0.0004;

gesture_len = 1;
window_len = 0.5;
step_len = 0.1;
num_win = floor((gesture_len - window_len)/step_len) + 1;

fs_emg = 2048;
Nsample_gesture = floor(fs_emg * gesture_len);
Nsample_window = floor(fs_emg * window_len);
Nsample_step = floor(fs_emg * step_len);

num_rest_predict_all = 0;
num_rest_real_all = 0;
num_gesture_predict_all = 0;
num_gesture_real_all = 0;


for num_choose = 1:length(index)
    ntime = 0;
    cd_time = 0;
    cd = 0;
    label1 = [];    label2 = [];    label = [];
    b = cell(1,10); a = cell(1,10);
    pointsMatrix = reshape(1:64,8,[])';
    badChannel_tmp = [];


    % wo = 50/(1000/2);
    % bw = wo/35;
    % [b1,a1] = iirnotch(wo,bw);
    % [b2,a2] = butter(4,[20 400]/500,'bandpass');

    [b{1},a{1}]= butter(8,10/(fs_emg/2),'high');
    [b{2},a{2}]= butter(8,500/(fs_emg/2),'low');
    [b{3},a{3}]= butter(4,[49.5,50.5]/(fs_emg/2),'stop');

    load([path_data,subject_test,num2str(session_test),'/model_nobd.mat']);

    % load([path_data,subject_test,num2str(session_test),'/data_flow.mat']);
    % load([path_data,subject_test,num2str(session_test),'/label_flow.mat']);

    load([path_data,subject_test,num2str(session_test),'/data_cut.mat']);
    load([path_data,subject_test,num2str(session_test),'/label_cut.mat']);
    data_flow = data_cut{1,num_choose};
    label_flow = label_cut{1,num_choose};

    n_model = length(mdl2);

    Npoints = size(data_flow,1);
    data_flow = double(data_flow);
    label_buffer = zeros(2,num_win);
    label_buffer_now = zeros(2,num_win);
    probability_mdl2_buffer_now = zeros(length(index),num_win);

    label1 = [];    label2 = [];
%     badChannel_buffer = zeros(size(data_flow,2),num_win);
%     badChannel_buffer_now = zeros(size(data_flow,2),num_win);

    while ntime < floor((Npoints-Nsample_window)/fs_emg) + 1
        ntime = ntime + step_len;
%         disp(ntime);
        id_time = int32(ntime/step_len);
        if ntime < window_len
            label1(id_time) = 0;
            label2(id_time) = 0;

            EMGdata_test = data_flow(floor(ntime*fs_emg)-Nsample_step+1:...
                floor(ntime*fs_emg),:);
            %     EMGdata_test_filter_tmp = filtfilt(b1,a1,EMGdata_test);
            %     emg = filtfilt(b2,a2,EMGdata_test_filter_tmp);
            data_preprocessed = EMGdata_test;
            for i = 1:3
                data_preprocessed = filtfilt(b{i},a{i},double(data_preprocessed));
            end
            emg = data_preprocessed;
%             badChannel_tmp = [badChannel_tmp,...
%                 my_badChannelDetection(emg,step_len,step_len,fs_emg)];
            continue
        end

%         if isempty(badChannel_tmp)
%             badChannel = [];
%         else
%             table_tmp = tabulate(badChannel_tmp);
%             table_tmp_tmp = table_tmp(:,2);
%             table_tmp_tmp(table_tmp_tmp>=2) = 1;
%             table_tmp_tmp(table_tmp_tmp<2) = 0;
%             badChannel_buffer(1:size(table_tmp_tmp,1),1:num_win) = ...
%                 repmat(table_tmp_tmp,1,num_win);
%         end


        if floor(ntime*fs_emg) >size(data_flow,1)
            break
        end
        EMGdata_test = data_flow(floor(ntime*fs_emg)-Nsample_window+1:...
            floor(ntime*fs_emg),:);
        %     EMGdata_test_filter_tmp = filtfilt(b1,a1,EMGdata_test);
        %     emg = filtfilt(b2,a2,EMGdata_test_filter_tmp);
        data_preprocessed = EMGdata_test;
        for i = 1:3
            data_preprocessed = filtfilt(b{i},a{i},double(data_preprocessed));
        end
        emg = data_preprocessed;
%         badChannel_tmp = my_badChannelDetection(emg,window_len,step_len,fs_emg);
%         table_tmp = tabulate(badChannel_tmp);
%         if isempty(table_tmp)
%             badChannel_buffer_now(:,1:num_win-1) = badChannel_buffer(:,2:num_win);
%             badChannel_buffer_now(:,num_win) = zeros(size(badChannel_buffer_now,1),1);
%             badChannel_buffer = badChannel_buffer_now;
%         else
%             badChannel_buffer_now(:,1:num_win-1) = badChannel_buffer(:,2:num_win);
%             badChannel_buffer_now(1:size(table_tmp,1),num_win) = table_tmp(:,2);
%             badChannel_buffer = badChannel_buffer_now;
%         end
%         badChannel_sum = sum(badChannel_buffer,2);
%         badChannel = find(badChannel_sum>floor(num_win/2));
%         badChannel = badChannel';

        rms = get_rms(emg,window_len,step_len,fs_emg);
        wl = get_wl(emg,window_len,step_len,fs_emg);
        zc = get_zc(emg,window_len,step_len,zc_ssc_thresh,fs_emg);
        ssc=get_ssc(emg,window_len,step_len,zc_ssc_thresh,fs_emg);

%         count = 0;
%         badChannel_tmp = badChannel;
%         while size(badChannel_tmp,2) > 0
%             badChannel_empty = [];
%             count = count + 1;
% %             if count > 2
% %                 fprintf('subjectID = %d, sessionID = %d, bad_channel_count > 2\n',...
% %                     subject_test_ID,session_test);
% %             end
%             for numBad = 1: size(badChannel_tmp,2)
%                 tempBad = badChannel_tmp(numBad);
%                 modTempBad = mod(tempBad,64);
%                 divTempBad = fix(tempBad/64);
%                 if modTempBad == 0
%                     modTempBad = 64;
%                     divTempBad = divTempBad - 1;
%                 end
%                 [col,row]=find(pointsMatrix==modTempBad);
%                 newColRow = [col-1,row;col+1,row;col,row-1;col,row+1];
%                 [errRow,errCol]=find(newColRow==0);
%                 if ~isempty(errRow)
%                     newColRow(errRow,:)=[];
%                 end
%                 [errRow2,errCol2]=find(newColRow==9);
%                 if ~isempty(errRow2)
%                     newColRow(errRow2,:)=[];
%                 end
%                 nnn=1;
%                 badChannelReplaceTemp = [];
%                 for n_new = 1:size(newColRow,1)
%                     badChannelReplaceTempCheck = pointsMatrix(newColRow(n_new,1),...
%                         newColRow(n_new,2))+divTempBad*64;
%                     if size( find(badChannel_tmp==badChannelReplaceTempCheck),2)==0
%                         badChannelReplaceTemp(nnn) = badChannelReplaceTempCheck;
%                         nnn=nnn+1;
%                     end
%                 end
%                 if isempty(badChannelReplaceTemp)
%                     badChannel_empty = [badChannel_empty,...
%                         badChannel_tmp(numBad)];
%                     continue
%                 end
% 
%             rms(1,badChannel_tmp(numBad))=mean(rms(1,badChannelReplaceTemp));
%             wl(1,badChannel_tmp(numBad))=mean(wl(1,badChannelReplaceTemp));
%             zc(1,badChannel_tmp(numBad))=mean(zc(1,badChannelReplaceTemp));
%             ssc(1,badChannel_tmp(numBad))=mean(ssc(1,badChannelReplaceTemp));
%             end
%             badChannel_tmp = badChannel_empty;
%           
%         end


        feature = [rms,wl,zc,ssc];
        feature_nor = L2Norm(feature);
        label_predict_tmp_1 = zeros(1,n_model);
        for i = 1:n_model
            if isempty(mdl1{1,i})
                label_predict_tmp_1(i) = 1;
                continue
            end
            mdl1_tmp = mdl1{1,i};
            %         mdl2_tmp = mdl2{1,i};
            feature_pca1 = feature_nor * P_pca1{1,i};
            [label_predict_tmp_1(i)] = my_cdem_multisource_test...
                (feature_pca1,mdl1_tmp);
            %         [label_predict_tmp_1(i)] = svmpredict(0,feature_pca1,mdl1_tmp);
        end

        %     [label_predict_tmp_1] = svmpredict(0,feature_nor,mdl1);


        %         if label_predict_tmp_1 == 1
        %             a=1;
        %         end



        sum_labels_1 = acc_target1*onehot(label_predict_tmp_1,2);
        [~,label_predict_tmp_1] = max(sum_labels_1,[],2);
        label_predict_tmp_1 = label_predict_tmp_1 - 1;


        label_buffer_now(:,1:num_win-1) = label_buffer(:,2:num_win);
        label_buffer_now(1,num_win) = label_predict_tmp_1;
        probability_mdl2_buffer_now(:,1:num_win-1) = probability_mdl2_buffer_now(:,2:num_win);
        [ges_mode_1,ges_times_1] = mode(label_buffer_now(1,:));
        if ges_times_1 > num_win - 3 && ges_mode_1 == 1
            label_predict_1 = 1;
        else
            label_predict_1 = 0;
        end
        label1(id_time) = label_predict_1;
        %         label1_now = label_predict_tmp_1;

        if label_predict_1 == 1 % motion:1; rest:0

            label_predict_tmp_tmp_2 = ones(1,n_model);
            for i = 1:n_model
                if isempty(mdl2{1,i})
                    label_predict_tmp_tmp_2(i) = 1; % because acc_target would be 0, label will not affetct the product
                    continue
                end
                mdl2_tmp = mdl2{1,i};
                feature_pca2 = feature_nor * P_pca2{1,i};
                %             feature_proj = feature_pca2 * P2{1,i};
                %             label_predict_tmp_tmp_2(i) = predict(mdl2_tmp,feature_proj);

                [label_predict_tmp_tmp_2(i)] = my_cdem_multisource_test...
                    (feature_pca2,mdl2_tmp);
            end

            sum_labels_2 = acc_target2*onehot(label_predict_tmp_tmp_2,length(index));
            sum_labels_2_nor = sum_labels_2/sum(sum_labels_2);

            
            probability_mdl2_buffer_now(:,num_win) = sum_labels_2_nor';
            sum_probability_mdl2_buffer_now = sum(probability_mdl2_buffer_now,2);
            [probability_tmp,label_predict_tmp_2] = max(sum_probability_mdl2_buffer_now);
            label_buffer_now(2,num_win) = label_predict_tmp_2;
            [ges_mode_2] = mode(label_buffer_now(2,:));

            if probability_tmp > thresh_classifier2 && ges_mode_2 ~= 0
                label_predict_2 = ges_mode_2;
            else
                label_predict_2 = 0;
            end
        else    % rest: 0
            label_predict_2 = 0;
            label_buffer_now(2,num_win) = 0;
            probability_mdl2_buffer_now(num_win) = 0;
        end

        label2(id_time) = label_predict_2;

        label_buffer = label_buffer_now;

        switch cd_mode
            case 'y'
                if cd == 1
                    label(id_time) = 0;
                    if ntime - cd_time >= 1
                        cd = 0;
                    end
                elseif cd == 0
                    if label_predict_2>0
                        label(id_time) = label_predict_2;
                        cd = 1;
                        cd_time = ntime;
                    else
                        label(id_time) = 0;
                    end
                end
            case 'n'
                label(id_time) = label_predict_2;
        end
    end



    tmp = label_flow;
    % tmp(tmp>0) = num_choose;
    for i = 1:length(index)
        tmp(tmp==index(i)) = i;
    end
    idx_tmp = floor(1:(fs_emg/10):length(tmp));
    idx_tmp = idx_tmp(1:length(label));
    label_real = tmp(idx_tmp);



    % gesture moment
    num_gesture_real = 0;
    num_gesture_predict = 0;
    cd_gesture_real = 0; % check whether label_real_now is gesture
    gesture_moment = [];    % 3 col: col1--start moment; col2--end moment;  col3--label_real
    cd_gesture_predict = 0; % check whether label_predict_now is gesture
    for i = 2:length(label_real)    % don't start during gesture, plz start during rest
        if cd_gesture_real == 0
            if label_real(i) > 0 && label_real(i-1) == 0    % 01
                cd_gesture_real = 1;
                num_gesture_real = num_gesture_real + 1;
                gesture_moment = [gesture_moment;[i,0,label_real(i)]];
            end
            % 00: do nothing
        else
            if label_real(i) == 0   % 10
                cd_gesture_real = 0;
                gesture_moment(end,2) = i;
                % 11:do nothing
            end
        end
    end

    % rest acc
    num_rest_real = 0;
    num_rest_predict = 0;
    
    for i = 1:length(label_real)
        if label_real(i) == 0
            num_rest_real = num_rest_real + 1;
            if label(i) == 0
                num_rest_predict = num_rest_predict + 1;
            end
        end
    end
    acc_rest(num_choose) = num_rest_predict / num_rest_real;

    % gesture acc
    for i = 1:size(gesture_moment,1)
        label_predict_tmp = label(gesture_moment(i,1):gesture_moment(i,2)+5);
        if sum(label_predict_tmp == gesture_moment(i,3)) > 0
            num_gesture_predict = num_gesture_predict +1;
        end
    end
    acc_gesture(num_choose) = num_gesture_predict / num_gesture_real;


    num_rest_predict_all = num_rest_predict_all + num_rest_predict;
    num_rest_real_all = num_rest_real_all + num_rest_real;
    num_gesture_predict_all = num_gesture_predict_all + num_gesture_predict;
    num_gesture_real_all = num_gesture_real_all + num_gesture_real;


%     figure(num_choose);
%     plot(label)
%     hold on
%     plot(label_real,'r');



end
acc_rest_all = num_rest_predict_all / num_rest_real_all;
acc_gesture_all = num_gesture_predict_all / num_gesture_real_all;

acc.acc_rest = acc_rest;
acc.acc_gesture = acc_gesture;
acc.acc_rest_all = acc_rest_all;
acc.acc_gesture_all = acc_gesture_all;


% plot(label2)
% hold on
% plot(label_real,'r');


% close all;
% label2_plot = label2;
% for i = length(index):-1:1
%     label2_plot(label2_plot==i) = index(i);
% end
% plot(label2_plot)
% hold on
% tmp = label_flow;
% idx_tmp = floor(1:(fs_emg/10):length(tmp));
% plot(tmp(idx_tmp),'r');

% close all;
% label2_plot = label2;
% plot(label2_plot)
% hold on
% tmp = label_flow;
% idx_tmp = floor(1:(fs_emg/10):length(tmp));
% plot(tmp(idx_tmp),'r');