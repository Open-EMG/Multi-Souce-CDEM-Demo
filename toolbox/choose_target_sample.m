function [Xt,Yt] = choose_target_sample(feature_test,...
    label_test,session_test,index,num_win)

feature = feature_test{1,session_test};
label = label_test{1,session_test};
t_index = []; 
for ii = 1:length(index)
    t_index_tmp = find(label==index(ii));
    if isempty(t_index_tmp)
        continue
    else
        t_index = [t_index;t_index_tmp(1:num_win)];
        t_index_tmp(1:num_win) = [];
    end
end
Xt = feature(t_index,:);
TempYt = label(t_index);
% X_test = feature(test_index,:);
% TempYtest = label(test_index);
% if source domain or target domain does not have all the
% gesture we want, continue
if size(unique(TempYt),1) < length(index)
    error('Gesture lost in target domain.');
end
for TempIndex = 1:length(index)
    placeYt = find(TempYt == index(TempIndex));
    Yt(placeYt) = TempIndex;
end

end

