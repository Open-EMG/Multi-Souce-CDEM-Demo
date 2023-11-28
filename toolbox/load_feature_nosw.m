function [feature_all] = load_feature_nosw(path_data,subjectID,sessionID, ...
    mode)

% mode:'motion' or 'rest'

num = length(subjectID)*length(sessionID);
feature_all = cell(3,num);
for i = 1:length(subjectID)
    for j = 1:length(sessionID)
        load([path_data,subjectID{i},sessionID{j},'/feature_',mode,...
            '_nosw_.mat'],['feature_',mode]);
        feature = eval(['feature_',mode,';']);
        feature_concat = [];
        for ii = 1:length(feature)
            feature_concat = [feature_concat;feature{1,ii}];
        end
        k = (i-1)*length(sessionID) + j;
        %         feature_all{1,k} = str2double(subjectID{i});
        %         label_all{1,k} = str2double(subjectID{i});
        feature_all{1,k} = i;
        feature_all{2,k} = str2double(sessionID{j});
        feature_all{3,k} = feature_concat;
    end
end

end

