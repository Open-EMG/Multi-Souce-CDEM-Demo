function [Xs,TempYs,id_sub,id_ses] = choose_source_sample(feature_train_all,...
    label_train_all,ID,index)
%CHOOSE_SOURCE_SAMPLE 此处显示有关此函数的摘要
%   此处显示详细说明
    iii = ID;
    feature_train = feature_train_all{3,iii};
    label_train = label_train_all{3,iii};
    id_sub = feature_train_all{1,iii};
    id_ses = feature_train_all{2,iii};
    s_index = find(ismember(label_train,index)==1);
    Xs = feature_train(s_index,:);
    TempYs = label_train(s_index);
end

