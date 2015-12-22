function [results] = main_cnn_C1_MP1_C2_MP2_FC1_FC2(TrnX, TrnY, TstX, TstY, params)
%%
%% input + conv1 + maxpool1 + conv2 + maxpool2 + fc1 + fc2 + output

lr              = params.lr;
momentum        = params.momentum;
maxepoches      = params.maxepoches;
trnbatchsize    = params.trnbatchsize;
tstbatchsize    = params.tstbatchsize;

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
H1 = params.H1; % image height
W1 = params.W1; % image width
B1 = params.B1; % image channels
F1 = params.F1; % C1: number of feature maps
h1 = params.h1; % C1: conv kernel height
w1 = params.w1; % C1: conv kernel width
k1      = cell(B1, F1); % C1: conv kernels
dk1     = cell(B1, F1); % C1: gradient of conv kernels
b1      = cell(1,  F1); % C1: bias
db1     = cell(1,  F1); % C1: gradient of bias
k1_old  = cell(B1, F1); % C1: conv kernels at last iteration
b1_old  = cell(1,  F1); % C1: bias at last iteration
dk1_old = cell(B1, F1); % C1: gradient of conv kernels at last iteration
db1_old = cell(1,  F1); % C1: gradient of bias at last iteration
for i=1:B1
    for j=1:F1
        k1{i,j} = normrnd(0,0.1, [h1, w1]); 
        b1{1,j} = 0;
        k1_old{i,j} = normrnd(0,0.1, [h1, w1]); 
        b1_old{1,j} = 0;
        dk1_old{i,j} = 0;
        db1_old{j} = 0;
    end
end
poolsize1 = params.poolsize1;
H2 = (H1-h1+1)/poolsize1;
W2 = (W1-w1+1)/poolsize1;
B2 = F1;
F2 = params.F2;
h2 = params.h2;
w2 = params.w2;
k2      = cell(B2, F2);
dk2     = cell(B2, F2);
b2      = cell(1,  F2);
db2     = cell(1,  F2);
k2_old  = cell(B2, F2);
b2_old  = cell(1,  F2);
dk2_old = cell(B2, F2);
db2_old = cell(1,  F2);
for i=1:B2
    for j=1:F2
        k2{i,j} = normrnd(0, 0.1, [h2, w2]);
        b2{1,j} = 0;
        k2_old{i,j} = normrnd(0, 0.1, [h2, w2]);
        b2_old{1,j} = 0;
        dk2_old{i,j} = 0;
        db2_old{1,j} = 0;
    end
end
poolsize2 = params.poolsize2;
% for FC1
n3 = (((H2-h2+1)/poolsize2)*((W2-w2+1)/poolsize2)) * F2;
n4 = params.n4;
n5 = size(TrnYM,1);
% normal distribution
W3 = normrnd(0,0.1,[n3, n4]); b3 = zeros(n4, 1);
W4 = normrnd(0,0.1,[n4, n5]); b4 = zeros(n5, 1);
dW4_old = 0;
db4_old = 0;
dW3_old = 0;
db3_old = 0;

loss_type = params.loss_type; %'cross-entropy';% '' or 'cross-entropy'
%% training
% lr = 0.00001;
% momentum = 0.0;
% maxepoches = 20;
TrnLoss = zeros(1, maxepoches);
TrnAccs = zeros(1, maxepoches);
TstAccs = zeros(1, maxepoches);

% for C1
figure(3),
kid = 1;
for i=1:B1
    for j=1:F1
        subplot(B1,F1,kid),imshow(k1{i,j},[])
        kid = kid + 1;
    end
end
drawnow;
% for C2
figure(4),
kid = 1;
for i=1:B2
    for j=1:F2
        subplot(B2,F2,kid),imshow(k2{i,j},[])
        kid = kid + 1;
    end
end
drawnow;

for epoch = 1:maxepoches
    
    %% testing
    TstY_pred = [];
    for tstbatch = 1:ntstbatches 
        s1 = (tstbatch-1)*tstbatchsize+1;
        s2 = tstbatch*tstbatchsize; 
        x = reshape(TstX(s1:s2, :), tstbatchsize, H1, W1, B1); %
        y = TstYM(:, s1:s2);
        
        % fprop
        % for I1-C1-MP1-I2
        a1 = cell(1, B1);
        for i=1:B1
           a1{i} = permute(x(:,:,:,i),[2,3,1]); % H1 x W1 * batchsize
        end
        zc1 = cell(1, F1);
        ac1 = cell(1, F1); 
        zp1 = cell(1, F1);
        ix1 = cell(1, F1);
        ap1 = cell(1, F1);
        for j=1:F1 
            zc1{j} = 0;
            for i=1:B1
                zc1{j} = zc1{j} + convn(a1{i}, k1{i,j}, 'valid');
            end
            zc1{j} = zc1{j} + b1{j};
            if 0    
                ac1{j} = sigmoid(zc1{j});
                [zp1{j}, ix1{j}] = MaxPooling(ac1{j}, [poolsize1, poolsize1]);
                ap1{j} = zp1{j}; %ap1{j} = sigmoid(zp1{j});
            elseif 0 
                ac1{j} = tanh_opt(zc1{j});
                [zp1{j}, ix1{j}] = MaxPooling(ac1{j}, [poolsize1, poolsize1]);
                ap1{j} = zp1{j}; %ap1{j} = tanh_opt(zp1{j});
            elseif 1 
                ac1{j} = relu(zc1{j});
                [zp1{j}, ix1{j}] = MaxPooling(ac1{j}, [poolsize1, poolsize1]);
                ap1{j} = zp1{j}; %ap1{j} = relu(zp1{j});
            end
        end
        % for I2-C2-MP2-I3
        a2 = cell(1, B2);
        for i=1:B2
           a2{i} = ap1{i}; 
        end
        zc2 = cell(1, F2);
        ac2 = cell(1, F2);
        zp2 = cell(1, F2);
        ix2 = cell(1, F2);
        ap2 = cell(1, F2);
        for j=1:F2
           zc2{j} = 0;
           for i=1:B2
               zc2{j} = zc2{j} + convn(a2{i}, k2{i,j}, 'valid');
           end
           zc2{j} = zc2{j} + b2{j};
           if 0
               ac2{j} = sigmoid(zc2{j});
               [zp2{j}, ix2{j}] = MaxPooling(ac2{j}, [poolsize2, poolsize2]);
               ap2{j} = zp2{j}; %ap2{j} = sigmoid(zp2{j});
           elseif 0
               ac2{j} = tanh_opt(zc2{j});
               [zp2{j}, ix2{j}] = MaxPooling(ac2{j}, [poolsize2, poolsize2]);
               ap2{j} = zp2{j}; %ap2{j} = tanh_opt(zp2{j});
           elseif 1
               ac2{j} = relu(zc2{j});
               [zp2{j}, ix2{j}] = MaxPooling(ac2{j}, [poolsize2, poolsize2]);
               ap2{j} = zp2{j}; %ap2{j} = relu(zp2{j});
           end
        end
        % for I3-FC1-O
        a3 = [];
        for j=1:F2
           a3 = cat(1, a3, reshape(ap2{j}, size(ap2{j}, 1)*size(ap2{j}, 2), size(ap2{j},3))); 
        end 
        z4 = W3'*a3 + repmat(b3, 1, size(a3, 2)); 
        if 0    % sigmoid
            a4 = sigmoid(z4);
        elseif 0% tanh
            a4 = tanh_opt(z4);
        elseif 1% relu
            a4 = relu(z4);
        end
        z5 = W4'*a4 + repmat(b4, 1, size(a4, 2));
        if strcmp(loss_type,'euclidean')
            if 0    % sigmoid
                a5 = sigmoid(z5);
            elseif 0% tanh
                a5 = tanh_opt(z5);
            elseif 1% relu
                a5 = relu(z5);
            end
        elseif strcmp(loss_type, 'cross-entropy')
            a5 = softmax(z5);                       
        end
        TstY_pred = cat(2, TstY_pred, a5); 
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
    TrnY_pred = [];
    for trnbatch = 1:ntrnbatches 
        s1 = (trnbatch-1)*trnbatchsize+1;
        s2 = trnbatch*trnbatchsize;
        
        x = reshape(TrnX(s1:s2, :), trnbatchsize, H1, W1, B1); %
        y = TrnYM(:, s1:s2);
        
        % fprop
        % for I1-C1-MP1-I2
        a1 = cell(1, B1);
        for i=1:B1
           a1{i} = permute(x(:,:,:,i),[2,3,1]); % H x W * batchsize
        end
        zc1 = cell(1, F1);
        ac1 = cell(1, F1); 
        zp1 = cell(1, F1);
        ix1 = cell(1, F1);
        ap1 = cell(1, F1);
        for j=1:F1 
            zc1{j} = 0;
            for i=1:B1
                zc1{j} = zc1{j} + convn(a1{i}, k1{i,j}, 'valid');
            end
            zc1{j} = zc1{j} + b1{j};
            if 0    
                ac1{j} = sigmoid(zc1{j});
                [zp1{j}, ix1{j}] = MaxPooling(ac1{j}, [poolsize1, poolsize1]);
                ap1{j} = zp1{j}; %ap1{j} = sigmoid(zp1{j});
            elseif 0 
                ac1{j} = tanh_opt(zc1{j});
                [zp1{j}, ix1{j}] = MaxPooling(ac1{j}, [poolsize1, poolsize1]);
                ap1{j} = zp1{j}; %ap1{j} = tanh_opt(zp1{j});
            elseif 1 
                ac1{j} = relu(zc1{j});
                [zp1{j}, ix1{j}] = MaxPooling(ac1{j}, [poolsize1, poolsize1]);
                ap1{j} = zp1{j}; %ap1{j} = relu(zp1{j});
            end
        end
        % for I2-C2-MP2-I3
        a2 = cell(1, B2);
        for i=1:B2
           a2{i} = ap1{i}; 
        end
        zc2 = cell(1, F2);
        ac2 = cell(1, F2);
        zp2 = cell(1, F2);
        ix2 = cell(1, F2);
        ap2 = cell(1, F2);
        for j=1:F2
           zc2{j} = 0;
           for i=1:B2
               zc2{j} = zc2{j} + convn(a2{i}, k2{i,j}, 'valid');
           end
           zc2{j} = zc2{j} + b2{j};
           if 0
               ac2{j} = sigmoid(zc2{j});
               [zp2{j}, ix2{j}] = MaxPooling(ac2{j}, [poolsize2, poolsize2]);
               ap2{j} = zp2{j}; %ap2{j} = sigmoid(zp2{j});
           elseif 0
               ac2{j} = tanh_opt(zc2{j});
               [zp2{j}, ix2{j}] = MaxPooling(ac2{j}, [poolsize2, poolsize2]);
               ap2{j} = zp2{j}; %ap2{j} = tanh_opt(zp2{j});
           elseif 1
               ac2{j} = relu(zc2{j});
               [zp2{j}, ix2{j}] = MaxPooling(ac2{j}, [poolsize2, poolsize2]);
               ap2{j} = zp2{j}; %ap2{j} = relu(zp2{j});
           end
        end
        % for I3-FC1-O
        a3 = [];
        for j=1:F2
           a3 = cat(1, a3, reshape(ap2{j}, size(ap2{j}, 1)*size(ap2{j}, 2), size(ap2{j},3))); 
        end 
        z4 = W3'*a3 + repmat(b3, 1, size(a3, 2));
        if 0    % sigmoid
            a4 = sigmoid(z4);
        elseif 0% tanh
            a4 = tanh_opt(z4);
        elseif 1% relu
            a4 = relu(z4);
        end
        z5 = W4'*a4 + repmat(b4, 1, size(a4, 2));
        if strcmp(loss_type,'euclidean')
            if 0    % sigmoid
                a5 = sigmoid(z5);
            elseif 0% tanh
                a5 = tanh_opt(z5);
            elseif 1% relu
                a5 = relu(z5);
            end
        elseif strcmp(loss_type, 'cross-entropy')
            a5 = softmax(z5);                       
        end
        TrnY_pred = cat(2, TrnY_pred, a5); 
        
        % bprop
        if strcmp(loss_type,'euclidean')
            delta_a5 = -(y - a5);  
            if 0        % sigmoid
                delta_z5 = delta_a5 .* a5 .* (1 - a5); 
            elseif 0    % tanh
                delta_z5 = delta_a5 .* (1.7159 * 2/3 * (1 - 1/(1.7159)^2 * a5.^2));
            elseif 1    % relu
                delta_z5 = delta_a5 .* double(a5 > 0);
            end
        elseif strcmp(loss_type, 'cross-entropy')
            delta_z5 = -(y - a5);                  
        end 
        % for FC2
        dW4 = (a4 * delta_z5') / size(a4, 2);
        db4 = mean(delta_z5, 2);
        % for FC1
        delta_a4 = W4 * delta_z5;
        if 0        % sigmoid
            delta_z4 = delta_a4 .* a4 .* (1 - a4);  
        elseif 0    % tanh
            delta_z4 = delta_a4 .* (1.7159 * 2/3 * (1 - 1/(1.7159)^2 * a4.^2));
        elseif 1    % relu
            delta_z4 = delta_a4 .* double(a4 > 0);
        end
        dW3 = (a3 * delta_z4') / size(a3, 2);
        db3 = mean(delta_z4, 2);
        
        delta_ap2 = W3 * delta_z4;       
        if 0        % sigmoid
            delta_zp2 = delta_ap2; %delta_zp2 = delta_ap2 .* a3 .* (1 - a3);  
        elseif 0    % tanh
            delta_zp2 = delta_ap2; %delta_zp2 = delta_ap2 .* (1.7159 * 2/3 * (1 - 1/(1.7159)^2 * a3.^2));
        elseif 1    % relu
            delta_zp2 = delta_ap2; %delta_zp2 = delta_ap2 .* double(a3 > 0);
        end
        % delta_a1 = W1 * delta_z2;
        % delta_z1 = delta_a1 .* a1 .* (ones(n1, size(a1, 2)) - a1);
        
        % delta_zp2 is the error sensitity of the layer after maxpooling
        % so at first, we should reshape them into the feature maps 
        delta_zp2 = permute(reshape(delta_zp2, (H2-h2+1)/poolsize2, (W2-w2+1)/poolsize2, F2, trnbatchsize), [1,2,4,3]);
        delta_ac2 = cell(1, F2);
        delta_zc2 = cell(1, F2);
        for j=1:F2
           delta_ac2{j} = zeros(H2-h2+1, W2-w2+1, trnbatchsize); 
           delta_ac2{j}(ix2{j}) = delta_zp2(:,:,:,j);
           if 0        % sigmoid
               delta_zc2{j} = delta_ac2{j}  .* ac2{j} .* (1 - ac2{j});
           elseif 0    % tanh
               delta_zc2{j} = delta_ac2{j}  .* (1.7159 * 2/3 * (1 - 1/(1.7159)^2 * ac2{j}.^2));
           elseif 1    % relu
               delta_zc2{j} = delta_ac2{j}  .* double(ac2{j} > 0);
           end 
           for i=1:B2 
               dk2{i,j} = 0;
               for jj=1:trnbatchsize 
                   dk2{i,j} = dk2{i,j} + imrotate(conv2(a2{i}(:,:,jj), imrotate(delta_zc2{j}(:,:,jj), 180), 'valid'), 180);
               end
               dk2{i,j} = dk2{i,j}/trnbatchsize;
               db2{j}   = sum(sum(reshape(delta_zc2{j}, size(delta_zc2{j},1)*size(delta_zc2{j},2), trnbatchsize), 1))/trnbatchsize;
           end
        end
        
        delta_a2 = cell(1, B2);
        for i=1:B2
           delta_a2{i} = 0;
           for j=1:F2
               delta_a2{i} = delta_a2{i} + convn(delta_zc2{j}, imrotate(k2{i,j}, 180), 'full');
           end
        end
        
        delta_ap1 = delta_a2;
        delta_zp1 = cell(1, F1);
        delta_ac1 = cell(1, F1);
        delta_zc1 = cell(1, F1);
        for j=1:F1
            if 0        % sigmoid
                delta_zp1{j} = delta_ap1{j}; %delta_zp1{j} = delta_ap1{j} .* ap1{j} .* (1 - ap1{j});
            elseif 0    % tanh
                delta_zp1{j} = delta_ap1{j}; %delta_zp1{j} = delta_ap1{j} .* (1.7159 * 2/3 * (1 - 1/(1.7159)^2 * ap1{j}.^2));
            elseif 1    % relu
                delta_zp1{j} = delta_ap1{j}; %delta_zp1{j} = delta_ap1{j} .* double(ap1{j} > 0);
            end
            delta_ac1{j} = zeros(H1-h1+1, W1-w1+1, trnbatchsize);
            delta_ac1{j}(ix1{j})=delta_zp1{j};
            if 0        % sigmoid
                delta_zc1{j} = delta_ac1{j} .* ac1{j} .* (1 - ac1{j});
            elseif 0    % tanh
                delta_zc1{j} = delta_ac1{j} .* (1.7159 * 2/3 * (1 - 1/(1.7159)^2 * ac1{j}.^2));
            elseif 1    % relu
                delta_zc1{j} = delta_ac1{j} .* double(ac1{j} > 0);
            end
            for i=1:B1 
               dk1{i,j} = 0;
               for jj=1:trnbatchsize 
                   dk1{i,j} = dk1{i,j} + imrotate(conv2(a1{i}(:,:,jj), imrotate(delta_zc1{j}(:,:,jj), 180), 'valid'), 180);
               end
               dk1{i,j} = dk1{i,j}/trnbatchsize;
               db1{j}   = sum(sum(reshape(delta_zc1{j}, size(delta_zc1{j},1)*size(delta_zc1{j},2), trnbatchsize), 1))/trnbatchsize;
           end
        end
        
        
        %% update parameters  
        % for C1-MP1
        for j=1:F1
            for i=1:B1
                dk1{i,j} = lr * dk1{i,j};
            end
            db1{j} = lr * db1{j};
        end
        % for C2-MP2
        for j=1:F2
            for i=1:B2
                dk2{i,j} = lr * dk2{i,j};
            end
            db2{j} = lr * db2{j};
        end
        % for FC1
        dW3 = lr * dW3; 
        db3 = lr * db3; 
        % for FC2
        dW4 = lr * dW4;
        db4 = lr * db4;
        
        if momentum > 0 
            % for C1-MP1
            for j=1:F1
                for i=1:B1
                    dk1_old{i,j} = dk1{i,j} + momentum * dk1_old{i,j}; dk1{i,j} = dk1_old{i,j};
                end
                db1_old{j} = db1{j} + momentum * db1_old{j}; db1{j} = db1_old{j};
            end
            % for C2-MP2
            for j=1:F2
                for i=1:B2
                    dk2_old{i,j} = dk2{i,j} + momentum * dk2_old{i,j}; dk2{i,j} = dk2_old{i,j};
                end
                db2_old{j} = db2{j} + momentum * db2_old{j}; db2{j} = db2_old{j};
            end
            % for FC1
           dW3_old = dW3 + momentum * dW3_old; dW3 = dW3_old;
           db3_old = db3 + momentum * db3_old; db3 = db3_old;
           % for FC2
           dW4_old = dW4 + momentum * dW4_old; dW4 = dW4_old;
           db4_old = db4 + momentum * db4_old; db4 = db4_old;
        end
        
        % update weights 
        % for C1-MP1
        for j=1:F1
            for i=1:B1
                k1{i,j} = k1{i,j} - dk1{i,j};
            end
            b1{j} = b1{j} - db1{j};
        end
        % for C2-MP2
        for j=1:F2
            for i=1:B2
                k2{i,j} = k2{i,j} - dk2{i,j};
            end
            b2{j} = b2{j} - db2{j};
        end
        % for FC1
        W3 = W3 - dW3;
        b3 = b3 - db3;
        % for FC2
        W4 = W4 - dW4;
        b4 = b4 - db4; 
        
    end   
    [~, TrnY_pred] = max(TrnY_pred, [], 1);
    TrnAccs(epoch) = length(find(TrnY_pred'==TrnY))/nTrn;
    fprintf('epoch=%02d,loss=%.6f, trn_acc=%.6f, tst_acc=%.6f\n', epoch, TrnLoss(epoch), TrnAccs(epoch), TstAccs(epoch));
    % plot training loss
    figure(1), plot(epoch, TrnLoss(epoch), 'r.'); hold on; drawnow
    % plot train accuracy and test accuracy
    figure(2), plot(epoch, TrnAccs(epoch), 'r.'); hold on; plot(epoch, TstAccs(epoch), 'b.'); hold on; drawnow
    % plot the C1 kernels
    figure(3), 
    kid = 1; 
    for i=1:B1
        for j=1:F1
           subplot(B1,F1,kid),imshow(k1{i,j},[])
           kid = kid + 1;
       end
    end
    drawnow;
    % plot the C2 kernels
    figure(4), 
    kid = 1;
    for i=1:B2
        for j=1:F2
           subplot(B2,F2,kid),imshow(k2{i,j},[])
           kid = kid + 1;
       end
    end
    drawnow;
    
    if mod(epoch, 10) ==0
       lr = lr / 10; 
    end
    
end

results = [];
results.TrnLoss = TrnLoss;
results.TrnAccs = TrnAccs;
results.TstAccs = TstAccs;
results.k1      = k1;
results.b1      = b1;
results.k2      = k2;
results.b2      = b2;
results.W3      = W3;
results.b3      = b3;
results.W4      = W4;
results.b4      = b4;


end


function a = sigmoid(z)
%% sigmoid function
% z: d x batchsize 
a = 1.0 ./ (1.0 + exp(-z));
end

function dz = sigmoid_d(z)
a = sigmoid(z);
dz = a .* (1 - a);
end

function a = tanh_opt(z)
%% hyperbolic tangent function
a = 1.7159 * tanh( 2/3 .* z);
end

function dz = tanh_opt_d(z)
%%  
a = tanh_opt(z);
dz = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * a.^2);
end

function a = softmax(z)
%% softmax function
% z: d x batchsize 
z = exp(bsxfun(@minus, z, max(z, [], 1)));
a = bsxfun(@rdivide, z, sum(z, 1));
end

function a = relu(z)
%% relu function
a = max(0, z);
end