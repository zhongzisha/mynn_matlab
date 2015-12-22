function [img_recon] = demo_patching(img, size_patch, size_skip, border)
%% 
% Note: This demo assumes gray-scale input image.
% for demo
if(nargin < 4), border = 'off'; end;
if(nargin < 3), size_skip = [3 3]; end;
if(nargin < 2), size_patch = [9 9]; end;
if(nargin < 1), img = im2double(imread('einstein.bmp')); end;
%%
size_img = size(img);
[patch] = im2patch(img, size_patch, size_skip, border);
[img_recon] = patch2im(patch, size_img, size_patch, size_skip, border);
imshow([img img_recon abs(img-img_recon)]);