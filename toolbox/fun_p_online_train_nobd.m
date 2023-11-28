function [] = fun_p_online_train_nobd(path_head,path_toolbox,path_data,subjectID,...
    sessionID,subject_test_ID,session_test,index)
%   method: knn,cdem

method = 'cdem';

addpath(genpath(path_toolbox));
addpath(genpath(path_head));

subject_test = subjectID{subject_test_ID};

num_win_motion = 10;
num_win_rest = 6;    % (len_gesture-len_win)/len_step+1=(1-0.5)/0.1+1=6

dim = 100;
window_len = 0.5;
step_len = 0.1;

feature_motion_all = load_feature_nobd(path_data,subjectID,sessionID, ...
    window_len,step_len,'motion');
feature_rest_all = load_feature_nobd(path_data,subjectID,sessionID, ...
    window_len,step_len,'rest');
label_all_motion = load_label(path_data,subjectID,sessionID,num_win_motion);
label_all_rest =  load_label(path_data,subjectID,sessionID,num_win_rest);

ID_test = find(ismember(subjectID,subject_test));
i = ID_test;

% for i=1:length(subjectID) % target person
[label_train_all_motion, label_test_motion] = split_label(label_all_motion,subjectID,sessionID,ID_test);
[label_train_all_rest, label_test_rest] = split_label(label_all_rest,subjectID,sessionID,ID_test);
[feature_motion_train_all,feature_motion_test] = split_feature(feature_motion_all,subjectID,sessionID,ID_test);
[feature_rest_train_all,feature_rest_test] = split_feature(feature_rest_all,subjectID,sessionID,ID_test);


j = session_test;
%     for j =1:2  % target domain & test data

[Xt_motion,Yt_motion] = choose_target_sample(feature_motion_test,...
    label_test_motion,session_test,index,num_win_motion);
[Xt_rest,Yt_rest] = choose_target_sample(feature_rest_test,...
    label_test_rest,session_test,index,num_win_rest);


acc_target1 = zeros(1,length(subjectID)*length(sessionID));
acc_target2 = zeros(1,length(subjectID)*length(sessionID));
mdl1 = cell(1,length(subjectID)*length(sessionID));
mdl2 = cell(1,length(subjectID)*length(sessionID));
% P2 = cell(1,length(subjectID)*length(sessionID));
P_pca1 = cell(1,length(subjectID)*length(sessionID));
P_pca2 = cell(1,length(subjectID)*length(sessionID));

%     Xt1 = [Xt_motion;Xt_rest];
%     Yt1 = [2*ones(size(Yt_motion)),ones(size(Yt_rest))];
% Xt1_nor = L2Norm(Xt1);
%     mdl1= svmtrain(Yt1',Xt1_nor,'-t 2');

for iii = 1:length(label_train_all_motion)  % source domain
    clear Ys_motion;
    [Xs_motion,TempYs,id_sub,id_ses] = choose_source_sample(...
        feature_motion_train_all,label_train_all_motion,iii,index);

    % if source domain or target domain does not have all the
    % gesture we want, continue
    if size(unique(TempYs),1)<length(index)
        continue;
    end
    for TempIndex = 1:length(index)
        placeYs = find(TempYs == index(TempIndex));
        Ys_motion(placeYs) = TempIndex;
    end

    [Xs_rest,TempYs_rest,~,~] = choose_source_sample(...
        feature_rest_train_all,label_train_all_rest,iii,index);
    Ys_rest = ones(size(TempYs_rest));

    %         domainS_features_ori = L2Norm(Xs')';
    %         domainT_features = L2Norm(Xt')';
    % Xs_gca =domainS_features_ori;Xt_gca=domainT_features;
    %
    %         domainS_features_ori = zscore(Xs')';
    %         domainT_features = zscore(Xt')';
    %         Xs_gca=mapminmax(full(domainS_features_ori' ))'; % Make sure that matrices are column and row normalized
    %         Xt_gca=mapminmax(full(domainT_features'))';

    Xs1 = [Xs_motion;Xs_rest];
    Ys1 = [2*ones(size(Ys_motion)),Ys_rest'];
    Xt1 = [Xt_motion;Xt_rest];
    Yt1 = [2*ones(size(Yt_motion)),ones(size(Yt_rest))];
    Xs2 = Xs_motion;
    Ys2 = Ys_motion;
    Xt2 = Xt_motion;
    Yt2 = Yt_motion;

    %     test_features = L2Norm(X_test);
    %     Xs_gca=zscore(full(domainS_features_ori)); % Make sure that matrices are column and row normalized
    %     Xt_gca=zscore(full(domainT_features));
    %     X_test_nor = zscore(full(test_features));
    %                 Xs_gca = Xs;
    %                 Xt_gca = Xt;
    %                 X_test_nor = X_test;
    Xs1_nor = L2Norm(Xs1);
    Xt1_nor = L2Norm(Xt1);
    Xs2_nor = L2Norm(Xs2);
    Xt2_nor = L2Norm(Xt2);

    %             options.lambda = 0.01;
    %             options.gamma = 0.001;
    %             options.beta = 0.001;
    %             options.eta = 0.0001;
    %             options.sigma=1;

    options.lambda = 1;
    options.gamma = 0.1;
    options.beta = 0.1;
    options.eta = 0.01;
    options.sigma=0.1;
    options.ReducedDim = dim;
    X1 = double([Xs1_nor;Xt1_nor]);
    X2 = double([Xs2_nor;Xt2_nor]);
    %                     options.ReducedDim = size(X,1)-1;
    P_pca1_tmp = PCA(X1,options);
    P_pca2_tmp = PCA(X2,options);
    P_pca1{1,((id_sub-1)*2+id_ses)} = P_pca1_tmp;
    P_pca2{1,((id_sub-1)*2+id_ses)} = P_pca2_tmp;
    Xs1_pca = Xs1_nor*P_pca1_tmp;
    Xt1_pca = Xt1_nor*P_pca1_tmp;
    Xs2_pca = Xs2_nor*P_pca2_tmp;
    Xt2_pca = Xt2_nor*P_pca2_tmp;

    %                     [acc_test((id_sub-1)*2+id_ses,(i-1)*2+j),correct{(id_sub-1)*2+id_ses,(i-1)*2+j},...
    %                         prob{(id_sub-1)*2+id_ses,(i-1)*2+j},acc_target((id_sub-1)*2+id_ses,(i-1)*2+j)] = ...
    %                         my_cdem_multisource_tmp(Xs_gca ,Ys,Xt_gca,Yt,test_features,Ytest,20,options);

    %             [acc_target((id_sub-1)*2+id_ses),mdl{1,(id_sub-1)*2+id_ses},]= ...
    %                 my_cdem_multisource_train(Xs2_nor ,Ys,Xt2_nor,Yt,20,options);


    %             mdl1{1,(id_sub-1)*2+id_ses}= svmtrain(Yt1',Xt1_nor,'-t 2');
    %             acc_target1((id_sub-1)*2+id_ses) = svmpredict(,feature_pca1,mdl1_tmp);
    [acc_target1((id_sub-1)*2+id_ses),mdl1{1,(id_sub-1)*2+id_ses}]= ...
        my_cdem_multisource_train(Xs1_pca ,Ys1,Xt1_pca,Yt1,20,options);

    [acc_target2((id_sub-1)*2+id_ses),mdl2{1,(id_sub-1)*2+id_ses}]= ...
        my_cdem_multisource_train(Xs2_pca ,Ys2,Xt2_pca,Yt2,20,options);

    %            [P_tmp]= my_cdem_multisource_train(Xs2_pca ,Ys2,Xt2_pca,Yt2,20,options);
    %            P2{1,(id_sub-1)*2+id_ses} = P_tmp;
    %            Xs2_proj = Xs2_pca * P_tmp;
    %            Xt2_proj = Xt2_pca * P_tmp;
    %             mdl2{1,(id_sub-1)*2+id_ses} = svmtrain([Ys2';Yt2'],[Xs2_proj;Xt2_proj],'-t 2');
    %             [~,acc_tmp,~] = svmpredict(Yt2',Xt2_proj,mdl2{1,(id_sub-1)*2+id_ses});
    %             acc_target2((id_sub-1)*2+id_ses) = acc_tmp(1);

    %                 acc(:,i*2+j-2) = CDEM(Xs_gca ,Ys,Xt_gca,Yt,...
    %                     20,1, options);


    clear Ys;


    %     fprintf('target_id = %d, source_id = %d\n',(i-1)*2+j,(id_sub-1)*2+id_ses);
end

save([path_data,subjectID{i},sessionID{j},'/model_nobd.mat'],...
    'mdl1','mdl2','acc_target1','acc_target2','P_pca1','P_pca2');

end