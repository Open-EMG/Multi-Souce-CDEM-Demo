
function [mdl,predLabels_t] = my_cdem_singlesource_train...
    (domainS_features,domainS_labels,...
    domainT_features,domainT_labels,d,options)
% num_iter = T;
options.ReducedDim = d;
options.alpha = 1;



% for iter = 1:num_iter
num_class = length(unique(domainS_labels));
W_all = zeros(size(domainS_features,1)+size(domainT_features,1));
W_s = constructW1(domainS_labels);
W = W_all;
W(1:size(W_s,1),1:size(W_s,2)) =  W_s;
% looping
% p = 1;
% predLabels = [];
% pseudoLabels = [];

% 计算P矩阵的function
P = constructP(domainS_features,domainS_labels,domainT_features,domainT_labels, W,options);
domainS_proj = domainS_features*P;
domainT_proj = domainT_features*P;
proj_mean_t = mean([domainS_proj;domainT_proj]);
domainS_proj = domainS_proj - repmat(proj_mean_t,[size(domainS_proj,1) 1 ]);
domainT_proj = domainT_proj - repmat(proj_mean_t,[size(domainT_proj,1) 1 ]);
domainS_proj = L2Norm(domainS_proj);
domainT_proj = L2Norm(domainT_proj);
%% distance to class means
classMeans = zeros(num_class,options.ReducedDim);
for i = 1:num_class
    classMeans(i,:) = mean(domainS_proj(domainS_labels==i,:));
end
classMeans = L2Norm(classMeans);
%     distClassMeans_t= EuDist2(domainT_proj,classMeans);
distClusterMeans_t= EuDist2(domainT_proj,classMeans);
% targetClusterMeans_t = vgg_kmeans(double(domainT_proj'), num_class, classMeans')';
% targetClusterMeans_t = L2Norm(targetClusterMeans_t);
% distClusterMeans_t = EuDist2(domainT_proj,targetClusterMeans_t);

%     expMatrix = exp(-distClassMeans);
expMatrix2_t = exp(-distClusterMeans_t);
%     probMatrix1 = expMatrix./repmat(sum(expMatrix,2),[1 num_class]);
probMatrix2_t = expMatrix2_t./repmat(sum(expMatrix2_t,2),[1 num_class]);

%     probMatrix = probMatrix1 * (1-iter./num_iter) + probMatrix2 * iter./num_iter;
probMatrix_t = probMatrix2_t;
[~,predLabels_t] = max(probMatrix_t,[],2);


% ----------------------------------------
%% calculate ACC
%     acc(iter) = sum(predLabels==test_labels)/length(test_labels);
%     probability{1,iter} = prob;
%     correct{1,iter} = predLabels == test_labels;

% acc_target = sum(predLabels_t'==domainT_labels)/length(domainT_labels);
mdl.P = P;
mdl.proj_mean = proj_mean_t;
mdl.classMeans = classMeans;
mdl.options = options;
mdl.num_class = num_class;


end