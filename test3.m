

N = 10;
H1 = 5;
W1 = 5;
h1 = 3;
w1 = 3;
B1 = 3;
F1 = 4;
a1 = rand(H1, W1, B1, N);
k1 = rand(h1, w1, B1, F1);
b1 = rand(1, F1);
OH1 = H1 - h1 + 1;
OW1 = W1 - w1 + 1;


%% conv by mm
a1_mat = zeros(OH1*OW1*N, h1*w1*B1);
for i=1:N
    for j=1:B1
        a1_mat(((i-1)*OH1*OW1+1):(i*OH1*OW1), ((j-1)*h1*w1+1):(j*h1*w1))=im2col(a1(:,:, j, i), [h1 w1], 'sliding')'; % (OH1*OW1) x (h1*w1)
    end
end
k1_mat = reshape(k1, h1*w1*B1, F1);
result1_mat = bsxfun(@plus, a1_mat * k1_mat, b1); % a1_mat: (OH1*OW1*N) x (h1*w1*B1), k1_mat: (h1*w1*B1) x (F1), b1: (1) x (F1)
result1 = permute(reshape(result1_mat, OH1, OW1, N, F1), [1 2 4 3]); % [OH1, OW1, F1, N]


%% conv by conv
result2 = zeros(OH1, OW1, F1, N);
for i=1:N
   for j=1:F1
       for l=1:B1
           result2(:,:,j,i) = result2(:,:,j,i) + filter2(k1(:,:,l,j), a1(:,:,l,i), 'valid');
       end
       result2(:,:,j,i) = result2(:,:,j,i) + b1(j);
   end
end





















