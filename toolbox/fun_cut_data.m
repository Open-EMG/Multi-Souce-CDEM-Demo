function [error] = fun_cut_data(path_head,path_toolbox,path_data,...
    subjectID,sessionID,index,subject_test_ID,session_test,fs_emg)
% path_head =  '/home/liujiayan/OTL_hyser/';
% path_toolbox = '/home/liujiayan/toolbox/';
addpath(genpath(path_toolbox));
addpath(genpath(path_head));

% path_data = '/home/liujiayan/v1_myv1/';
% subjectID ={'DCY','JXY','WXP','WMQ','WYM','LYW','DY','PS','LXY',...
%     'FJH','GY','CQ','DHK','GYP','ML','TLK','YJX','ZGY','RHR','DHQ'};
% sessionID = {'1','2'};

% index = [6,7,8,9,10,11,30,31,32,34];
% index = [2,3,5,6,9,10,15,20,21,24];

% subject_test_ID = 1;
% session_test = 2;
subject_test = subjectID{subject_test_ID};
% fs_emg = 2048;

load([path_data,subject_test,num2str(session_test),'/data_flow.mat']);
load([path_data,subject_test,num2str(session_test),'/label_flow.mat']);
% load([path_data,subject_test,num2str(session_test),'/model.mat']);

% path_save = [path_data,subject_test,num2str(session_test),'/data_cut/'];
% if exist(path_save,'dir')~=7
%     mkdir(path_save);
% end

data_cut = cell(1,length(index));
label_cut = cell(1,length(index));
error = 0;

for i = 1:length(index)
    id_tmp = find(label_flow == index(i));
    if isempty(id_tmp)
        subject_test_ID
        session_test
        error = 1;
        break
    end
    % id_start: should skip the first gesture sample in data stream
    tmp = find(diff(id_tmp)>1);
    id_start = id_tmp(tmp(1)+1);
    id_end = id_tmp(end);
    data_cut{1,i} = data_flow(id_start-1.5*fs_emg:id_end+2*fs_emg,:);
    label_cut{1,i} = label_flow(id_start-1.5*fs_emg:id_end+2*fs_emg);
end

if error == 0
save([path_data,subject_test,num2str(session_test),'/data_cut.mat'],'data_cut');
save([path_data,subject_test,num2str(session_test),'/label_cut.mat'],'label_cut');
end
end