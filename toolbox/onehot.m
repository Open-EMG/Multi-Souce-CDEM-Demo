function [label_onehot] = onehot(label_raw,num_label)
% label_raw: colunm vector, integer 1-n (n = num_label)
% num_label: max label

label_onehot = zeros(length(label_raw),num_label);
for i = 1:length(label_raw)
    label_onehot(i,label_raw(i)) = 1;
end
