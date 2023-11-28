function [label_train_all, label_test] = split_label(label_all,subjectID,sessionID,ID_test)

% for i=1:length(subjectID) % target person

label_train_all = label_all;   label_test = cell(1,length(sessionID));
k_test = [];
for k = 1:length(subjectID)*length(sessionID)
    if label_all{1,k} == ID_test
        label_test{1,label_all{2,k}} = label_all{3,k};
        k_test = [k_test,k];
    end
end
label_train_all(:,k_test) = [];
