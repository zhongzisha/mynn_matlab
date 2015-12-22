

clear;

N = 10;
C = 3;
H = 5;
W = 5;
M = 2;
K = 3;

OH = H - K + 1;
OW = W - K + 1;

% unfold input x
x = rand(H, W, C, N);
x_mat = [];
for i=1:N
    xxx = [];
    for j=1:C
       xxx = cat(2, xxx, im2col(x(:,:, j, i), [K K], 'sliding')'); 
    end
    x_mat = cat(1, x_mat, xxx);
end
% x_mat: ((H-K+1)*(W-K+1)*N, K*K*C)

% unfold conv filters
k = rand(K, K, C, M);
if 0
    k_mat = [];
    for i=1:M
        xxx=[];
        for j=1:C
            xxx = cat(1, xxx, im2col(k(:,:,j,i), [K K], 'sliding'));
        end
        k_mat = cat(2, k_mat, xxx);
    end
else
    k_mat = reshape(k, K*K*C, M);
end

% do conv
y_mat = x_mat * k_mat;

% reshape output y
y_mat = reshape(y_mat, OH, OW, N, M);
y = permute(y_mat, [1 2 4 3]);

% the following is coventional convolution
y1 = zeros(OH, OW, M, N);
for i=1:N
    for j=1:M
        fr = 0;
        for l=1:C
            % the following two is equivalent, both are correlation
            % fr = fr + conv2(x(:,:,l,i),imrotate(k(:,:,l,j),180),'valid');
            fr = fr + filter2(k(:,:,l,j), x(:,:,l,i),'valid');
        end
        y1(:,:,j,i) = fr;
    end
end















