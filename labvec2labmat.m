function YM = labvec2labmat(Y)
% Y: n x 1

[~,indexes]=ismember(Y,unique(Y));
c=length(indexes);
rows = 1:length(Y); %// row indx
YM = zeros(length(Y),length(unique(indexes))); %// A matrix full of zeros
YM(sub2ind(size(YM),rows ,indexes')) = 1; %// Ones at the desired row/column combinations
YM=sparse(YM);























