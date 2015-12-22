function img = mergePatch(patches, h1, w1, H, W)

img = zeros(H, W);
p_idx = 1;
for ww=1:w1
    for hh=1:h1
        pp=col2im(patches(:, p_idx), [h1 w1], [H W], 'sliding');
        img(hh:hh+H-h1, ww:ww+W-w1) = pp;
        p_idx = p_idx + 1;
    end
end



