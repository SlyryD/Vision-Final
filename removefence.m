imgs_rm = imgs;

im_to_change_idx = 110;

% Difference between pixels within which they are considered "the same"
Tdiff = 20;

% Keep track of the average flows
avg = zeros(50, 1);

for k = 50:-1:1
    % Check if this is the image we're working with
    is_result = k == 50;
    
    im = flows{im_to_change_idx - 10 + k};
    next_im = imgs{im_to_change_idx - 9 + k};
    mask = ones(size(im, 1), size(im, 2));
    cur_im = imgs{im_to_change_idx - 10 + k};
    
    if is_result
        main = cur_im;
    end
    
%     magflow = sqrt(im(:, :, 1).^2 + im(:, :, 2).^2);
    magflow = im(:, :, 1);
    
    avg(k) = mean2(magflow);

    T = avg(k) * 1.11;

    for i = 1:size(im, 1)
        for j = 1:size(im, 2)
            if magflow(i, j) < T
                mask(i, j) = 0;
            end
        end
    end
    
    % Go through all non-masked pixels and look where they should be next
    % frame based on their flow
    if ~is_result
        pixels = find(mask)';
        for i = 1:size(pixels, 1);
            [x, y] = ind2sub(size(mask), pixels(i));

            flow = magflow(x, y);
            
            % Adjust flow for different average flow in the next frame
            flow = (flow / avg(k)) * avg(k + 1);
            
            expected_color = cur_im(x, y, :);
            seen_color = next_im(x, y + uint32(flow), :);
            
            ec = reshape(expected_color, 3, 1);
            sc = reshape(seen_color, 3, 1);
            
            if norm(double(ec - sc)) < Tdiff
                % If colors are different, assume the fence got in the way
                 main(x, y + uint32(flow), :) = seen_color;
            end
        end
    else
        maxmag = max(max(magflow));
    end
        
    size(mask)
%     imshow(magflow / maxmag)
     imshow(main);
    
    imgs_rm{k} = cur_im;
end
    