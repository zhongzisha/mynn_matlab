load /media/slave1data/common_data/mnist/mnist/mnist_test.mat
load /media/slave1data/common_data/mnist/mnist/mnist_train.mat

lr = 0.01;
momentum = 0.9;
maxepoches = 90;

if 0
    %% I + FC1 + FC2 + O
    params              = [];
    params.lr           = 0.01;  
    params.momentum     = 0.9;
    params.maxepoches   = 50;
    params.trnbatchsize = 100;
    params.tstbatchsize = 1000;
    params.loss_type    = 'cross-entropy'; % 'euclidean'
    params.n2           = 500;
    params.n3           = 500;
    [results] = main_nn_FC1_FC2(train_X,train_labels, test_X, test_labels, params); 
end

if 0
    %% I + C1 + MP1 + FC1 + O
    params              = [];
    params.lr           = 0.01;  
    params.momentum     = 0.9;
    params.maxepoches   = 50;
    params.trnbatchsize = 100;
    params.tstbatchsize = 1000;
    params.H1           = 28;
    params.W1           = 28;
    params.B1           = 1; % number of channels
    params.F1           = 4; % number of feature maps
    params.h1           = 5;
    params.w1           = 5;
    params.poolsize1    = 2;
    params.loss_type    = 'cross-entropy'; % 'euclidean'
    [results] = main_cnn_C1_MP1_FC1(train_X,train_labels, test_X, test_labels, params); 
end


if 0
    %% I + C1 + MP1 + C2 + MP2 + FC1 + O
    params              = [];
    params.lr           = 0.01;  
    params.momentum     = 0.9;
    params.maxepoches   = 5;
    params.trnbatchsize = 100;
    params.tstbatchsize = 1000;
    params.H1           = 28;
    params.W1           = 28;
    params.B1           = 1; % number of channels
    params.F1           = 4; % number of feature maps
    params.h1           = 5;
    params.w1           = 5;
    params.poolsize1    = 2;
    params.F2           = 4;
    params.h2           = 3;
    params.w2           = 3;
    params.poolsize2    = 2;
    params.loss_type    = 'cross-entropy'; % 'euclidean'
    [results] = main_cnn_C1_MP1_C2_MP2_FC1(train_X,train_labels, test_X, test_labels, params); 
end

if 1
    %% I + C1 + MP1 + C2 + MP2 + FC1 + FC2 + O
    params              = [];
    params.lr           = 0.01;  
    params.momentum     = 0.9;
    params.maxepoches   = 5;
    params.trnbatchsize = 20;
    params.tstbatchsize = 100;
    params.loss_type    = 'cross-entropy'; % 'euclidean'
    params.H1           = 28; % I
    params.W1           = 28; % I
    params.B1           = 1;  % I
    params.F1           = 4;  % C1
    params.h1           = 5;  % C1
    params.w1           = 5;  % C1
    params.poolsize1    = 2;  % MP1
    params.F2           = 6;  % C2
    params.h2           = 5;  % C2
    params.w2           = 5;  % C2
    params.poolsize2    = 2;  % MP2
    params.n4           = 500;% FC1
    [results]= main_cnn_C1_MP1_C2_MP2_FC1_FC2(train_X, train_labels, test_X, test_labels, params);
    figure,
    plot(results.TrnLoss);
    figure,
    plot(results.TrnAccs, 'r');
    hold on, 
    plot(results.TstAccs, 'b');
end