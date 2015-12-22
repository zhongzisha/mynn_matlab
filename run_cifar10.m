
clear all;
close all;

TrnX = [];
TrnY = [];
for i=1:5
    load(sprintf('/media/slave1data/common_data/cifar-10-batches-mat/data_batch_%d.mat', i));
    TrnX = cat(1, TrnX, double(data));
    TrnY = cat(1, TrnY, double(labels)+1);
end
load /media/slave1data/common_data/cifar-10-batches-mat/test_batch.mat
TstX = double(data);
TstY = double(labels)+1;
clear data labels

if 0
    %% I + FC1 + FC2 + O
    params = [];
    params.lr           = 0.01;  
    params.momentum     = 0.9;
    params.maxepoches   = 10;
    params.trnbatchsize = 100;
    params.tstbatchsize = 1000;
    params.loss_type    = 'cross-entropy'; % 'euclidean'
    params.n2           = 1000;
    params.n3           = 500;
    [results] = main_nn_FC1_FC2(TrnX,TrnY, TstX, TstY, params);
    figure,
    plot(results.TrnLoss);
    figure,
    plot(results.TrnAccs, 'r');
    hold on, 
    plot(results.TstAccs, 'b');
    save('main_nn_FC1_FC2.mat','params','results');
end

if 0
    %% I + C1 + MP1 + FC1 + O
    params              = [];
    params.lr           = 0.01;  
    params.momentum     = 0.9;
    params.maxepoches   = 20;
    params.trnbatchsize = 50;
    params.tstbatchsize = 100;
    params.loss_type    = 'cross-entropy'; % 'euclidean'
    params.H1           = 32; % I
    params.W1           = 32; % I
    params.B1           = 3;  % I
    params.F1           = 8; % C1
    params.h1           = 5;  % C1
    params.w1           = 5;  % C1
    params.poolsize1    = 2;  % MP1
    [results] = main_cnn_C1_MP1_FC2(TrnX,TrnY, TstX, TstY, params);
    figure,
    plot(results.TrnLoss);
    figure,
    plot(results.TrnAccs, 'r');
    hold on, 
    plot(results.TstAccs, 'b');
    save('main_cnn_C1_MP1_FC2.mat','params','results');
end


if 0
    %% I + C1 + MP1 + FC1 + O
    params              = [];
    params.lr           = 0.01;  
    params.momentum     = 0.9;
    params.maxepoches   = 90;
    params.trnbatchsize = 50;
    params.tstbatchsize = 100;
    params.loss_type    = 'cross-entropy'; % 'euclidean'
    params.H1           = 32; % I
    params.W1           = 32; % I
    params.B1           = 3;  % I
    params.F1           = 8; % C1
    params.h1           = 5;  % C1
    params.w1           = 5;  % C1
    params.poolsize1    = 2;  % MP1
    [results] = main_cnn_C1_MP1_FC2_speedup(TrnX,TrnY, TstX, TstY, params);
    figure,
    plot(results.TrnLoss);
    figure,
    plot(results.TrnAccs, 'r');
    hold on, 
    plot(results.TstAccs, 'b');
    save('main_cnn_C1_MP1_FC2_speedup.mat','params','results');
end

if 0
    %% I + C1 + MP1 + C2 + MP2 + FC3 + O
    params              = [];
    params.lr           = 0.01;  
    params.momentum     = 0.9;
    params.maxepoches   = 20;
    params.trnbatchsize = 50;
    params.tstbatchsize = 1000;
    params.loss_type    = 'cross-entropy'; % 'euclidean'
    params.H1           = 32; % I
    params.W1           = 32; % I
    params.B1           = 3;  % I
    params.F1           = 8;  % C1
    params.h1           = 5;  % C1
    params.w1           = 5;  % C1
    params.poolsize1    = 2;  % MP1
    params.F2           = 8;  % C2
    params.h2           = 5;  % C2
    params.w2           = 5;  % C2
    params.poolsize2    = 2;  % MP2
    [results] = main_cnn_C1_MP1_C2_MP2_FC3(TrnX,TrnY, TstX, TstY, params);
    figure,
    plot(results.TrnLoss);
    figure,
    plot(results.TrnAccs, 'r');
    hold on, 
    plot(results.TstAccs, 'b');
    save('main_cnn_C1_MP1_C2_MP2_FC3.mat','params','results');
end

if 0
    %% I + C1 + MP1 + C2 + MP2 + FC3 + O
    params              = [];
    params.debug        = 0;
    params.lr           = 0.01;  
    params.lr_stepsize  = 20;
    params.momentum     = 0.9;
    params.maxepoches   = 90;
    params.trnbatchsize = 400;
    params.tstbatchsize = 100;
    params.loss_type    = 'cross-entropy'; % 'euclidean'
    params.H1           = 32; % I
    params.W1           = 32; % I
    params.B1           = 3;  % I
    params.F1           = 8;  % C1
    params.h1           = 5;  % C1
    params.w1           = 5;  % C1
    params.poolsize1    = 2;  % MP1
    params.F2           = 16;  % C2
    params.h2           = 5;  % C2
    params.w2           = 5;  % C2
    params.poolsize2    = 2;  % MP2
    [results] = main_cnn_C1_MP1_C2_MP2_FC3_speedup(TrnX, TrnY, TstX, TstY, params);
%     figure,
%     plot(results.TrnLoss);
%     figure,
%     plot(results.TrnAccs, 'r');
%     hold on, 
%     plot(results.TstAccs, 'b');
    save('main_cnn_C1_MP1_C2_MP2_FC3_speedup.mat','params','results');
end


if 1
    %% I + C1 + MP1 + C2 + MP2 + FC3 + O, parallel
    params              = [];
    params.debug        = 0;
    params.lr           = 0.001;  
    params.lr_stepsize  = 20;
    params.momentum     = 0.9;
    params.maxepoches   = 90;
    params.trnbatchsize = 100;
    params.tstbatchsize = 100;
    params.loss_type    = 'cross-entropy'; % 'euclidean'
    params.H1           = 32; % I
    params.W1           = 32; % I
    params.B1           = 3;  % I
    params.F1           = 8;  % C1
    params.h1           = 5;  % C1
    params.w1           = 5;  % C1
    params.poolsize1    = 2;  % MP1
    params.F2           = 16;  % C2
    params.h2           = 5;  % C2
    params.w2           = 5;  % C2
    params.poolsize2    = 2;  % MP2
    params.num_workers  = 4;  % number of workers for training
    [results] = main_cnn_C1_MP1_C2_MP2_FC3_speedup_dataparallel(TrnX, TrnY, TstX, TstY, params);
%     figure,
%     plot(results.TrnLoss);
%     figure,
%     plot(results.TrnAccs, 'r');
%     hold on, 
%     plot(results.TstAccs, 'b');
    save('main_cnn_C1_MP1_C2_MP2_FC3_speedup_dataparallel.mat','params','results');
end

if 0
    %% I + C1 + MP1 + C2 + MP2 + FC1 + FC2 + O
    params              = [];
    params.lr           = 0.01;  
    params.momentum     = 0.9;
    params.maxepoches   = 50;
    params.trnbatchsize = 100;
    params.tstbatchsize = 1000;
    params.loss_type    = 'cross-entropy'; % 'euclidean'
    params.H1           = 32; % I
    params.W1           = 32; % I
    params.B1           = 3;  % I
    params.F1           = 16; % C1
    params.h1           = 5;  % C1
    params.w1           = 5;  % C1
    params.poolsize1    = 2;  % MP1
    params.F2           = 16; % C2
    params.h2           = 5;  % C2
    params.w2           = 5;  % C2
    params.poolsize2    = 2;  % MP2
    params.n4           = 256;% FC1
    [results]= main_cnn_C1_MP1_C2_MP2_FC1_FC2(TrnX, TrnY, TstX, TstY, params);
    figure,
    plot(results.TrnLoss);
    figure,
    plot(results.TrnAccs, 'r');
    hold on, 
    plot(results.TstAccs, 'b');
    save('results_cnn_c1_mp1_c2_mp2_fc1_fc2.mat','params','results');
end