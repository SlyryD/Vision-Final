imgs_rm = imgs;

threshold = 1.11;

for k = 1:20
    im = flows{k};
    im_rm = imgs_rm{k};
    
    % Get the magnitude of the flow
    magflow = real(im).^2 + imag(im).^2;
 
    avg = mean2(magflow);
    T = avg * threshold;

    for i = 1:size(im, 1)
        for j = 1:size(im, 2)
            % Use the magnitude of the flow to determine if the pixels
            % should remain
            if magflow(i, j) < T
                magflow(i, j) = 0;
            else
                im_rm(i, j, 1) = 0;
                im_rm(i, j, 2) = 0;
                im_rm(i, j, 3) = 0;
            end
        end
    end
    
    if k == 1
        maxmag = max(max(magflow));
    end
    
    imshow(im_rm);
    
    imwrite(im_rm, sprintf('.\\result\\result_frame_%d_rgb.png', k + 80))
    
    imgs_rm{k} = im_rm;
end
    