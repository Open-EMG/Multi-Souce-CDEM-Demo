function [mdl1,mdl2,acc_target1,acc_target2,P_pca1_tmp,P_pca2_tmp] = ...
    fun_online_train(Xs_motion,Xs_rest,Ys_motion,Ys_rest,...
    Xt_motion,Xt_rest,Yt_motion,Yt_rest)

% num_size = 40;  % v1: 20*2

dim = 100;

Xs1 = [Xs_motion;Xs_rest];
Ys1 = [2*ones(size(Ys_motion)),Ys_rest'];
Xt1 = [Xt_motion;Xt_rest];
Yt1 = [2*ones(size(Yt_motion)),ones(size(Yt_rest))];
Xs2 = Xs_motion;
Ys2 = Ys_motion;
Xt2 = Xt_motion;
Yt2 = Yt_motion;

Xs1_nor = L2Norm(Xs1);
Xt1_nor = L2Norm(Xt1);
Xs2_nor = L2Norm(Xs2);
Xt2_nor = L2Norm(Xt2);

options.lambda = 1;
options.gamma = 0.1;
options.beta = 0.1;
options.eta = 0.01;
options.sigma=0.1;
options.ReducedDim = dim;
X1 = double([Xs1_nor;Xt1_nor]);
X2 = double([Xs2_nor;Xt2_nor]);

P_pca1_tmp = PCA(X1,options);
P_pca2_tmp = PCA(X2,options);
P_pca1_tmp = single(P_pca1_tmp);
P_pca2_tmp = single(P_pca2_tmp);
Xs1_pca = Xs1_nor*P_pca1_tmp;
Xt1_pca = Xt1_nor*P_pca1_tmp;
Xs2_pca = Xs2_nor*P_pca2_tmp;
Xt2_pca = Xt2_nor*P_pca2_tmp;

[acc_target1,mdl1]= ...
    my_cdem_multisource_train(Xs1_pca ,Ys1,Xt1_pca,Yt1,20,options);

[acc_target2,mdl2]= ...
    my_cdem_multisource_train(Xs2_pca ,Ys2,Xt2_pca,Yt2,20,options);

clear Ys;

end