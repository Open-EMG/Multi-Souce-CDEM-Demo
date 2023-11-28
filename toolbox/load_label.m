function [label_all] = load_label(path_data,subjectID,sessionID,num_win)
num = length(subjectID)*length(sessionID);
label_all = cell(3,num);
for i = 1:length(subjectID)
    for j = 1:length(sessionID)
        label_motion = load([path_data,subjectID{i},sessionID{j},'/label_motion.mat']);
        label = label_motion.label_motion;
        for ii = 1:length(label)
            label_repeat_tmp = repmat(label,1,num_win);
            label_repeat = reshape(label_repeat_tmp',[numel(label_repeat_tmp),1]);
        end
        k = (i-1)*length(sessionID) + j;
        %         feature_all{1,k} = str2double(subjectID{i});
        %         label_all{1,k} = str2double(subjectID{i});
        label_all{1,k} = i;
        label_all{2,k} = str2double(sessionID{j});
        label_all{3,k} = label_repeat;
    end
end

end

