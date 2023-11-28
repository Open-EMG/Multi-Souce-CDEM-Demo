function [feature_train_all,feature_test] = split_feature(feature_all,subjectID,sessionID,ID_test)
feature_train_all = feature_all; feature_test = cell(1,length(sessionID));
k_test = [];
for k = 1:length(subjectID)*length(sessionID)
    if feature_all{1,k} == ID_test
        feature_test{1,feature_all{2,k}} = feature_all{3,k};
        k_test = [k_test,k];
    end
end
feature_train_all(:,k_test) = [];


end

