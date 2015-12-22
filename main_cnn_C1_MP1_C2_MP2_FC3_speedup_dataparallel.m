function [results] = main_cnn_C1_MP1_C2_MP2_FC3_speedup_dataparallel(TrnX, TrnY, TstX, TstY, params)
%%
%% input + conv1 + maxpool1 + conv2 + maxpool2 + fc3 + output

lr              = params.lr;
lr_stepsize     = params.lr_stepsize;
momentum        = params.momentum;
maxepoches      = params.maxepoches;
trnbatchsize    = params.trnbatchsize;
tstbatchsize    = params.tstbatchsize;
debug           = params.debug;

num_workers     = params.num_workers;
parallel_step   = trnbatchsize / num_workers; 

if num_workers > 1
    if 0
        poolobj = parpool(num_workers);
    else
        matlabpool(num_workers);
    end 
end

nTrn = size(TrnX, 1);
TrnYM = full(labvec2labmat(TrnY))'; % nTrn x c
nTst = size(TstX, 1);
TstYM = full(labvec2labmat(TstY))'; % nTst x c

%% mean normalization
TrnX_mean = mean(TrnX, 1);
TrnX = TrnX - repmat(TrnX_mean, nTrn, 1);
TstX = TstX - repmat(TrnX_mean, nTst, 1);
%% intensity/127.5-1.0
% TrnX = (TrnX*2)-1.0;
% TstX = (TstX*2)-1.0;
%% mean-std normalization
% TrnX_mean = mean(TrnX, 1);
% TrnX_std  = max(std(TrnX),eps);
% TrnX = bsxfun(@minus, TrnX, TrnX_mean);
% TrnX = bsxfun(@rdivide, TrnX, TrnX_std);
% TstX = bsxfun(@minus, TstX, TrnX_mean);
% TstX = bsxfun(@rdivide, TstX, TrnX_std);

%%
rndidx = randperm(nTrn);
TrnX = TrnX(rndidx, :);
TrnYM = TrnYM(:, rndidx);
TrnY = TrnY(rndidx);
ntrnbatches = floor(nTrn/trnbatchsize);
ntstbatches = floor(nTst/tstbatchsize);

%% initialization
H1 = params.H1;
W1 = params.W1;
B1 = params.B1; % number of channels
F1 = params.F1; % number of feature maps
h1 = params.h1;
w1 = params.w1;
OH1 = H1 - h1 + 1;
OW1 = W1 - w1 + 1;
k1      = normrnd(0, 0.0001, [h1, w1, B1, F1]);
b1      = zeros(1, F1);
dk1_old = zeros(h1, w1, B1, F1);
db1_old = zeros(1, F1);

poolsize1 = params.poolsize1;
H2 = OH1/poolsize1;
W2 = OW1/poolsize1;
B2 = F1;
F2 = params.F2;
h2 = params.h2;
w2 = params.w2;
OH2 = H2 - h2 + 1;
OW2 = W2 - w2 + 1;
k2      = normrnd(0, 0.01, [h2, w2, B2, F2]); 
b2      = zeros(1, F2); 
dk2_old = zeros(h2, w2, B2, F2);
db2_old = zeros(1, F2);

poolsize2 = params.poolsize2;
H3 = OH2 / poolsize2;
W3 = OW2 / poolsize2;
% for FC1
n3 = (H3 * W3) * F2;
n4 = size(TrnYM,1);
% normal distribution
Weight3 = normrnd(0,0.01,[n3, n4]); 
bias3 = zeros(n4, 1);
dWeight3_old = 0;
dbias3_old = 0;
loss_type = params.loss_type; %'cross-entropy';% '' or 'cross-entropy'

TrnLoss = zeros(1, maxepoches);
TrnAccs = zeros(1, maxepoches);
TstAccs = zeros(1, maxepoches);
TrnTime = zeros(1, maxepoches);

if debug
    hh=figure;
    kid = 1;
    for i=1:B1
        for j=1:F1
            subplot(B1,F1,kid),imshow(k1(:,:,i,j),[])
            kid = kid + 1;
        end
    end
    drawnow;
end

for epoch = 1:maxepoches
    
    %% testing
    TstY_pred = [];
    for tstbatch = 1:ntstbatches 
        
        %% prepare mini-batch data
        s1 = (tstbatch-1)*tstbatchsize+1;
        s2 = tstbatch*tstbatchsize; 
        x = reshape(TstX(s1:s2, :), tstbatchsize, H1, W1, B1); %
        y = TstYM(:, s1:s2);
        
        %% fprop
        % fprop-I1
        N = tstbatchsize;
        a1 = permute(x, [2,3,4,1]);
        % fprop-C1
        a1_mat = zeros(OH1*OW1*N, h1*w1*B1);
        for i=1:N 
            for j=1:B1
                 a1_mat(((i-1)*OH1*OW1+1):(i*OH1*OW1), ((j-1)*h1*w1+1):(j*h1*w1))=im2col(a1(:,:, j, i), [h1 w1], 'sliding')'; % (OH1*OW1) x (h1*w1)
            end 
        end
        k1_mat = reshape(k1, h1*w1*B1, F1);
        zc1_mat = bsxfun(@plus, a1_mat * k1_mat, b1); % a1_mat: (OH1*OW1*N) x (h1*w1*B1), k1_mat: (h1*w1*B1) x (F1), b1: (1) x (F1)
        zc1 = permute(reshape(zc1_mat, OH1, OW1, N, F1), [1 2 4 3]); % [OH1, OW1, F1, N]
        
        % fprop-C1-neuron
        if 0
            ac1 = sigmoid(zc1);
        elseif 0
            ac1 = tanh_opt(zc1);
        elseif 1
            ac1 = relu(zc1);
        end
        
        % fprop-MP1 
        [zp1, ix1] = MaxPooling(ac1, [poolsize1, poolsize1]);
        ap1 = zp1;
        
        % fprop-I2
        a2 = ap1; % H2 x W2 x F1 x N
        % fprop-C2
        a2_mat = zeros(OH2*OW2*N, h2*w2*B2);
        for i=1:N 
            for j=1:F1
                a2_mat(((i-1)*OH2*OW2+1):(i*OH2*OW2), ((j-1)*h2*w2+1):(j*h2*w2)) = im2col(a2(:,:, j, i), [h2 w2], 'sliding')';
            end 
        end
        k2_mat = reshape(k2, h2*w2*B2, F2);
        zc2_mat = bsxfun(@plus, a2_mat * k2_mat, b2); % a2_mat: (OH2*OW2*N) x (h2*w2*B2), k2_mat: (h2*w2*B2) x (F2), b2: (1) x (F2)
        zc2 = permute(reshape(zc2_mat, OH2, OW2, N, F2), [1 2 4 3]); % [OH2, OW2, F2, N]
        % fprop-C2-neuron
        if 0
            ac2 = sigmoid(zc2);
        elseif 0
            ac2 = tanh_opt(zc2);
        elseif 1
            ac2 = relu(zc2);
        end
        % fprop-MP2 
        [zp2, ix2] = MaxPooling(ac2, [poolsize2, poolsize2]);
        ap2 = zp2;
        
        % I3
        a3 = reshape(ap2, n3, N);
        % FC3
        z4 = bsxfun(@plus, Weight3'*a3, bias3);  % n4 x N
        
        % Output
        if strcmp(loss_type,'euclidean')
            if 0     % sigmoid
                a4 = sigmoid(z4);
            elseif 0 % tanh
                a4 = tanh_opt(z4);
            elseif 0 % linear
                a4 = z4;
            elseif 1 % relu
                a4 = relu(z4);
            end
        elseif strcmp(loss_type, 'cross-entropy')
            a4 = softmax(z4);                       
        end
        TstY_pred = cat(2, TstY_pred, a4); 
    end
    
    if strcmp(loss_type, 'euclidean') 
        error = TstYM - TstY_pred;
        TrnLoss(epoch) = 0.5 * trace(error'*error) / nTrn;
    elseif strcmp(loss_type, 'cross-entropy')
        TrnLoss(epoch) = - sum(sum(TstYM .* log(TstY_pred))) / nTrn;
    end
    [~, TstY_pred] = max(TstY_pred, [], 1);
    TstAccs(epoch) = length(find(TstY_pred'==TstY))/nTst; 
    
    %% training
    tic;
    TrnY_pred = zeros(n4, nTrn);
    for trnbatch = 1:ntrnbatches 
        s1 = (trnbatch-1)*trnbatchsize+1;
        s2 = trnbatch*trnbatchsize;
        
        x = reshape(TrnX(s1:s2, :), trnbatchsize, H1, W1, B1); %
        y = TrnYM(:, s1:s2);
        
        % here, split the batch training data into several parts, and
        % dispatch them into workers to compute the gradients
        spmd(num_workers) 
            s11 = (labindex - 1)*parallel_step + 1;
            s21 = labindex * parallel_step;
            xx = x(s11:s21,:,:,:);
            yy = y(:,s11:s21);
            [pred_b, dk1_b, db1_b, dk2_b, db2_b, dWeight3_b, dbias3_b] = main_cnn_C1_MP1_C2_MP2_FC3_speedup_dataparallel_slave(xx,yy,params, k1, b1, k2, b2, Weight3, bias3);
        end
        
        dk1 = 0; db1 = 0;
        dk2 = 0; db2 = 0;
        dWeight3 = 0; dbias3 = 0;
        pred = [];
        for i=1:num_workers
           temp= dk1_b(i);  dk1 = dk1 + temp{1};
           temp= db1_b(i);  db1 = db1 + temp{1};
           temp= dk2_b(i);  dk2 = dk2 + temp{1};
           temp= db2_b(i);  db2 = db2 + temp{1};
           temp= dWeight3_b(i);  dWeight3 = dWeight3 + temp{1};
           temp= dbias3_b(i);  dbias3 = dbias3 + temp{1};
           temp = pred_b(i); pred = cat(2, pred, temp{1});
        end 
        TrnY_pred(:, s1:s2) = pred;
        
        %% compute gradient update parameters  
        % for C1
        dk1 = momentum * dk1_old + lr * dk1 / trnbatchsize; dk1_old = dk1;
        db1 = momentum * db1_old + lr * db1 / trnbatchsize; db1_old = db1;
        % for C2
        dk2 = momentum * dk2_old + lr * dk2 / trnbatchsize; dk2_old = dk2;
        db2 = momentum * db2_old + lr * db2 / trnbatchsize; db2_old = db2;
        % for FC3
        dWeight3 = momentum * dWeight3_old + lr * dWeight3 / trnbatchsize; dWeight3_old = dWeight3;
        dbias3 = momentum * dbias3_old + lr * dbias3 / trnbatchsize; dbias3_old = dbias3;
        
        % update weights 
        % for C1
        k1 = k1 - dk1;
        b1 = b1 - db1;
        % for C2
        k2 = k2 - dk2;
        b2 = b2 - db2;
        % for FC3
        Weight3 = Weight3 - dWeight3;
        bias3 = bias3 - dbias3;
        
    end   
    TrnTime(epoch) = toc;
    [~, TrnY_pred] = max(TrnY_pred, [], 1);
    TrnAccs(epoch) = length(find(TrnY_pred'==TrnY))/nTrn;
    fprintf('epoch=%02d,loss=%.6f, trn_acc=%.6f, tst_acc=%.6f, time=%.6f\n', epoch, TrnLoss(epoch), TrnAccs(epoch), TstAccs(epoch), TrnTime(epoch));
    
    if mod(epoch, lr_stepsize) ==0
       lr = lr / lr_stepsize; 
    end
    
    if debug
        figure(hh),
        kid = 1;
        for i=1:B1
            for j=1:F1
                subplot(B1,F1,kid),imshow(k1(:,:,i,j),[])
                kid = kid + 1;
            end
        end
        drawnow;
    end
end

results = [];
results.TrnLoss = TrnLoss;
results.TrnAccs = TrnAccs;
results.TstAccs = TstAccs;
results.TrnTime = TrnTime;
results.k1      = k1;
results.b1      = b1;
results.k2      = k2;
results.b2      = b2;
results.Weight3      = Weight3;
results.bias3      = bias3; 

if num_workers > 1
    if 0
        delete(poolobj);
    else
        matlabpool close
    end
end

end