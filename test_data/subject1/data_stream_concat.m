% Not necessary
% Concat data together into a cell
% & concat label together into a cell

clear;clc;close all
data_cut_tmp = cell(1,10);
label_cut_tmp = cell(1,10);
for i = 1:10
    load(['data_subject1_',num2str(i),'.mat']);
    load(['label_subject1_',num2str(i),'.mat']);
    data_cut_tmp{i} = data_cut;
    label_cut_tmp{i} = label_cut;
end
data_cut = data_cut_tmp;
label_cut = label_cut_tmp;
save('data_cut.mat','data_cut');
save('label_cut.mat','label_cut');