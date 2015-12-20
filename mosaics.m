function [] = mosaics()
% Initialize
run('C:\\vlfeat-0.9.20\\toolbox\\vl_setup');

% Read in images
idxs = 101:366;
n = length(idxs);
imgs = cell(n, 1);
for i = 1:n
    imgs{i} = imread(sprintf('.\\images\\rgb\\frame_%d_rgb.png', idxs(i)));
end

% Get windows for each frame
gap = 30;
windows = cell(n, 1);
for i = 1:n
    window = i:gap:(i + 3*gap);
    while max(window) > n
        window = window - gap;
    end
    windows{i} = window(window ~= i);
end

% Get homographies
% Hs = cell(n, 3);
% for i = 1:n
%     window = windows{i};
%     for j = 1:length(window)
%         k = window(j);
%         fprintf('Get homography %d -> %d\n', i, k);
%         [H, Fm] = getH(imgs{i}, imgs{k});
%         Hs{i, j} = H;
%     end
% end
% save('homographies', 'Hs');
load('homographies');

% % Compute temporal median filter for window
% M = 7;
% i = 15;
% frame(:, :, :, M + 1) = double(imgs{i});
% [X, Y, Z] = size(frame);
% window = (i - M):(i + M);
% window = window(window ~= i);
% Fm = zeros(size(Fms{1}));
% for j = window
%     if j < i
%         Fm = Fm + Fms{j, i - j};
%     else
%         Fm = Fm + Fms{i, j - i};
%     end
% end
% for j = window
%     if j < i
%         H = Hs{j, i - j};
%     else
%         H = inv(Hs{i, j - i});
%     end
%     
%     ur = 1:Y;
%     vr = 1:X;
%     [u,v] = meshgrid(ur,vr);
%     z_ = H(3,1) * u + H(3,2) * v + H(3,3);
%     u_ = (H(1,1) * u + H(1,2) * v + H(1,3)) ./ z_;
%     v_ = (H(2,1) * u + H(2,2) * v + H(2,3)) ./ z_;
%     
%     k = M + j - i + 1;
%     other = imgs{j};
%     tmp = vl_imwbackward(im2double(other), u_, v_);
%     tmp(isnan(tmp)) = 0;
%     frame(:, :, :, k) = tmp;
% end
% imshow(frame);
% imwrite(frame, sprintf('.\\result\\result_frame_%d_rgb.png', idxs(i)))

% [buddies, Hs] = getH(imgs, 19);
% save('homographies', 'buddies', 'Hs');
% load('homographies')

% Remove fence from images
imgs_rm = cell(n, 1);
for i = 1:n
    imgs_rm{i} = imread(sprintf('.\\imgs_rm\\rm_frame_%d_rgb.png', idxs(i)));
end

for i = 1:n
    frame = imgs_rm{i};
    window = windows{i};
    for j = 1:length(window)
        k = idxs(window(j));
        fprintf('Stitching frames %d and %d\n', idxs(i), k)
        H = sift_mosaic(frame, imgs{k});
        frame = stitch(frame, imgs{k}, H);
    end
%     fprintf('Stitching frames %d and %d\n', idxs(i), idxs(window(2)))
%     H = sift_mosaic(frame, imgs{idxs(window(2))});
%     frame = stitch(frame, imgs{idxs(window(2))}, H);
    figure(3);
    imshow(frame);
    drawnow;
    imwrite(frame, sprintf('.\\result\\result_frame_%d_rgb.png', idxs(i)));
end

% % Compute optical flows
% opticalFlow = vision.OpticalFlow('OutputValue', 'Horizontal and vertical components in complex form');
% flows = cell(n, 1);
% for i = 1:(n)
%     flows{i} = step(opticalFlow, single(rgb2gray(imgs{i})));
% end

% current = imgs{1};
% for i = 4:3:19
%     current = sift_mosaic(current, imgs{i});
%     imshow(current);
%     drawnow;
% end

% % Mosaic
% windowsize = 3;
% % windowsize = 21;
% figure(1); imshow(imgs_rm{1}); drawnow;
% figure(2); imshow(imgs_rm{19}); drawnow;
% result = stitchframe(imgs_rm, 1, windowsize);
% imshow(result);

% % Stitch images
% for i = 1:n
%     Hs{i} = Hs{1};
% end
% fprintf(1, 'Stitching...\n');
% for i = 1:n
%     frame = stitch(imgs_rm{i}, imgs_rm{buddies(i)}, Hs{i});
%     imwrite(frame, sprintf('.\\result\\result_frame_%d_rgb.png', idxs(i)));
% end

% for i = 1:n
%     frame = stitchframe(imgs, i, [5, 5, 3]);
%     imwrite(frame, sprintf('.\\result\\result_frame_%d_rgb.png', idxs(i)));
% end
end

function [H, Fm] = getH(frame, other)
% Convert to grayscale
grayframe = rgb2gray(frame);
grayother = rgb2gray(other);

% Compute optical flow between frames
opticalFlow = vision.OpticalFlow('ReferenceFrameSource', 'Input port', ...
    'OutputValue', 'Horizontal and vertical components in complex form');
V = step(opticalFlow, single(grayframe), single(grayother));

% Compute projection vector
u = real(V);
v = imag(V);
centeredV = V - mean(mean(V));
uprime = real(centeredV);
vprime = imag(centeredV);

% Project each optical flow vector
M = u.*uprime./vprime + v;
M = max(M, -M);
thetah = prctile(M(:), 90);
thetal = prctile(M(:), 10);

% Compute significance of each pixel
Fm = (max(min(M, thetah), thetal) - thetal)/(thetah - thetal);
[X1, X2] = sift_matches(grayframe, grayother);

% Compute homography using least squares
W = ones(size(X2, 2), 1);
for k = 1:size(X2, 2)
    W(k) = 1 - Fm(sub2ind(size(M), round(X2(2, k)), round(X2(1, k))));
end
H = lscov(X2', X1', W)';
end

function H = sift_mosaic(im1, im2)
% SIFT_MOSAIC Demonstrates matching two images using SIFT and RANSAC
%
%   SIFT_MOSAIC demonstrates matching two images based on SIFT
%   features and RANSAC and computing their mosaic.
%
%   SIFT_MOSAIC by itself runs the algorithm on two standard test
%   images. Use SIFT_MOSAIC(IM1,IM2) to compute the mosaic of two
%   custom images IM1 and IM2.

% AUTORIGHTS

% make single
im1 = im2single(im1) ;
im2 = im2single(im2) ;

% make grayscale
if size(im1,3) > 1, im1g = rgb2gray(im1) ; else im1g = im1 ; end
if size(im2,3) > 1, im2g = rgb2gray(im2) ; else im2g = im2 ; end

% --------------------------------------------------------------------
%                                                         SIFT matches
% --------------------------------------------------------------------

[f1,d1] = vl_sift(im1g) ;
[f2,d2] = vl_sift(im2g) ;

[matches, scores] = vl_ubcmatch(d1,d2) ;

numMatches = size(matches,2) ;

X1 = f1(1:2,matches(1,:)) ; X1(3,:) = 1 ;
X2 = f2(1:2,matches(2,:)) ; X2(3,:) = 1 ;

% --------------------------------------------------------------------
%                                         RANSAC with homography model
% --------------------------------------------------------------------

clear H score ok ;
for t = 1:100
  % estimate homograpyh
  subset = vl_colsubset(1:numMatches, 4) ;
  A = [] ;
  for i = subset
    A = cat(1, A, kron(X1(:,i)', vl_hat(X2(:,i)))) ;
  end
  [~,S,V] = svd(A) ;
  H{t} = reshape(V(:,9),3,3) ;

  % score homography
  X2_ = H{t} * X1 ;
  du = X2_(1,:)./X2_(3,:) - X2(1,:)./X2(3,:) ;
  dv = X2_(2,:)./X2_(3,:) - X2(2,:)./X2(3,:) ;
  ok{t} = (du.*du + dv.*dv) < 6*6 ;
  score(t) = sum(ok{t}) ;
end

[score, best] = max(score) ;
H = H{best} ;
ok = ok{best} ;

% --------------------------------------------------------------------
%                                                  Optional refinement
% --------------------------------------------------------------------

function err = residual(H)
 u = H(1) * X1(1,ok) + H(4) * X1(2,ok) + H(7) ;
 v = H(2) * X1(1,ok) + H(5) * X1(2,ok) + H(8) ;
 d = H(3) * X1(1,ok) + H(6) * X1(2,ok) + 1 ;
 du = X2(1,ok) - u ./ d ;
 dv = X2(2,ok) - v ./ d ;
 err = sum(du.*du + dv.*dv) ;
end

if exist('fminsearch')
  H = H / H(3,3) ;
  opts = optimset('Display', 'none', 'TolFun', 1e-8, 'TolX', 1e-8) ;
  H(1:8) = fminsearch(@residual, H(1:8)', opts) ;
else
  warning('Refinement disabled as fminsearch was not found.') ;
end

% --------------------------------------------------------------------
%                                                         Show matches
% --------------------------------------------------------------------

% dh1 = max(size(im2,1)-size(im1,1),0) ;
% dh2 = max(size(im1,1)-size(im2,1),0) ;
% 
% figure(1) ; clf ;
% subplot(2,1,1) ;
% imagesc([padarray(im1,dh1,'post') padarray(im2,dh2,'post')]) ;
% o = size(im1,2) ;
% line([f1(1,matches(1,:));f2(1,matches(2,:))+o], ...
%      [f1(2,matches(1,:));f2(2,matches(2,:))]) ;
% title(sprintf('%d tentative matches', numMatches)) ;
% axis image off ;
% 
% subplot(2,1,2) ;
% imagesc([padarray(im1,dh1,'post') padarray(im2,dh2,'post')]) ;
% o = size(im1,2) ;
% line([f1(1,matches(1,ok));f2(1,matches(2,ok))+o], ...
%      [f1(2,matches(1,ok));f2(2,matches(2,ok))]) ;
% title(sprintf('%d (%.2f%%) inliner matches out of %d', ...
%               sum(ok), ...
%               100*sum(ok)/numMatches, ...
%               numMatches)) ;
% axis image off ;
% drawnow ;

% x1 = f1(1:2,matches(1,ok)); x1(3, :) = 1;
% x2 = f2(1:2,matches(2,ok)); x2(3, :) = 1;
% H = (x1' \ x2')';

% mosaic = mosaic_helper(im1, im2, H);
end

function mo = mosaic_helper(im1, im2, H)
% --------------------------------------------------------------------
%                                                               Mosaic
% --------------------------------------------------------------------

box2 = [1  size(im2,2) size(im2,2)  1 ;
        1  1           size(im2,1)  size(im2,1) ;
        1  1           1            1 ] ;
box2_ = H \ box2 ;
box2_(1,:) = box2_(1,:) ./ box2_(3,:) ;
box2_(2,:) = box2_(2,:) ./ box2_(3,:) ;
ur = min([1 box2_(1,:)]):max([size(im1,2) box2_(1,:)]) ;
vr = min([1 box2_(2,:)]):max([size(im1,1) box2_(2,:)]) ;

[u,v] = meshgrid(ur,vr) ;
im1_ = vl_imwbackward(im2double(im1),u,v) ;

z_ = H(3,1) * u + H(3,2) * v + H(3,3) ;
u_ = (H(1,1) * u + H(1,2) * v + H(1,3)) ./ z_ ;
v_ = (H(2,1) * u + H(2,2) * v + H(2,3)) ./ z_ ;
im2_ = vl_imwbackward(im2double(im2),u_,v_) ;

% im1_(isnan(im1_)) = 0;
% im2_(isnan(im2_)) = 0;
% [Y, X, Z] = size(im1_);
% for x = 1:X
%     for y = 1:Y
%         if all(im1_(y, x, :) < 1/64) && any(im2_(y, x, :) >= 1/64)
%             im1_(y, x, :) = im2_(y, x, :);
%         end
%     end
% end
% mosaic = im1_;

mass = ~isnan(im1_) + ~isnan(im2_) ;
im1_(isnan(im1_)) = 0 ;
im2_(isnan(im2_)) = 0 ;
mo = (im1_ + im2_) ./ mass ;

figure(3) ; clf ;
imagesc(mo) ; axis image off ;
title('Mosaic') ;
end

function [X1, X2] = sift_matches(im1, im2)
% make single
im1 = im2single(im1) ;
im2 = im2single(im2) ;

% make grayscale
if size(im1,3) > 1, im1g = rgb2gray(im1) ; else im1g = im1 ; end
if size(im2,3) > 1, im2g = rgb2gray(im2) ; else im2g = im2 ; end

% --------------------------------------------------------------------
%                                                         SIFT matches
% --------------------------------------------------------------------

[f1,d1] = vl_sift(im1g) ;
[f2,d2] = vl_sift(im2g) ;

[matches, scores] = vl_ubcmatch(d1,d2) ;

numMatches = size(matches,2) ;

X1 = f1(1:2,matches(1,:)) ; X1(3,:) = 1 ;
X2 = f2(1:2,matches(2,:)) ; X2(3,:) = 1 ;
end

function frame = stitchframe(images, idx, windowsize)
% Get window around frame
n = length(images);
frame = images{idx};
gray = single(rgb2gray(frame));
st = floor(windowsize(3));
window = (idx - st):(idx + st);
window = window(window >= 1 & window <= n);
window = window(window ~= idx);

% Compute optical flow between frame and each neighbor
for j = window
    % Compute optical flow between frames
    opticalFlow = vision.OpticalFlow('ReferenceFrameSource', 'Input port', 'OutputValue', 'Horizontal and vertical components in complex form');
    V = step(opticalFlow, gray, single(rgb2gray(images{j})));
    
    % Compute projection vector
    u = real(V);
    v = imag(V);
    centeredV = V - mean(mean(V));
    uprime = real(centeredV);
    vprime = imag(centeredV);
    
    % Project each optical flow vector
    M = u.*uprime./vprime + v;
    M = max(M, -M);
    thetah = prctile(M(:), 90);
    thetal = prctile(M(:), 10);
    
    % Compute significance of each pixel
    Fm = (max(min(M, thetah), thetal) - thetal)/(thetah - thetal);
    [X1, X2] = sift_matches(gray, rgb2gray(images{j}));
    
    % Compute homography using least squares
    W = ones(size(X2, 2), 1);
    for k = 1:size(X2, 2)
        W(k) = 1 - Fm(sub2ind(size(M), round(X2(2, k)), round(X2(1, k))));
    end
    A = lscov(X2', X1', W)';
    
    % Stitch together
    frame = stitch(frame, images{j}, V);
end

% % Get SIFT features for frame
% singleframe = single(rgb2gray(frame));
% [fframe, dframe] = vl_sift(singleframe);
% figure(3);
% imshow(frame);
% hframe = vl_plotframe(fframe);
% set(hframe, 'color', 'r', 'linewidth', 1);

% % Stitch images with frame
% k = window(1) - 1;
% singles = cell(windowsize - 1, 1);
% fs = cell(windowsize - 1, 1);
% ds = cell(windowsize - 1, 1);
% for j = window
%     % Convert to single grayscale matrices
%     l = j - k;
%     fprintf(1, 'Creating mosaic with frame %d and %d\n', idx, j);
%     singles{l} = single(rgb2gray(images{j}));
%     
%     % Get sift features
%     fprintf(1, 'SIFT...\n');
%     [fs{l}, ds{l}] = vl_sift(singles{l});
%     figure(4);
%     imshow(images{22});
%     himage = vl_plotframe(fs{1});
%     set(himage, 'color', 'r', 'linewidth', 1);
    
%     % RANSAC
%     fprintf(1, 'RANSAC...\n');
%     A = ransac(fframe, dframe, fs{l}, ds{l});
% 
%     % Stitch images
%     fprintf(1, 'Stitching...\n');
%     frame = stitch(frame, images{j}, A);
%     figure(3);
%     imshow(frame);
%     drawnow;
% end
end

function bestA = ransac(f1, d1, f2, d2)
% Initialize
warning('off', 'MATLAB:singularMatrix');
warning('off', 'MATLAB:nearlySingularMatrix');
quality = 0;
bestA = eye(3);
matches = find_matches(f1, d1, f2, d2);
numMatches = size(matches, 2);
for k = 1:1e4
    % Find 3 random pairs of matching points
    perm = randperm(numMatches);
    sel = perm(1:3);
    rand_matches = matches(:, sel); % 3 random matches
    p2 = rand_matches(1:2, :); % Points in LR2
    p1 = rand_matches(5:6, :); % Points in LR1

    % Compute affine transformation
    A = affine_transformation(p1, p2);    
    if rcond(A) < 1e-10
        continue;
    end

    % Count number of close features after transformation
    count = 0;
    for i = 1:numMatches
        p = matches(1:2, i); q = matches(5:6, i);
        if norm(A*[p; 1] - [q; 1]) < 2
            count = count + 1;
        end
    end

    % Update best transform
    if count > quality
        quality = count;
        bestA = A;
    end
end
end

function score = ssd(d1, d2)
score = sum((d1 - d2).^2);
end

function matches = find_matches(f1, d1, f2, d2)
X = size(f1, 2); Y = size(f2, 2);
matches = Inf*ones(9, Y);
for j = 1:Y
    matches(1:4, j) = f2(1:4, j);
    for i = 1:X
        s = ssd(d1(:, i), d2(:, j));
        if s < matches(9, j)
            matches(5:8, j) = f1(1:4, i);
            matches(9, j) = s;
        end
    end
end
end

function transform = affine_transformation(p1, p2)
transform = [p1; ones(1, 3)] / [p2; ones(1, 3)]; % Leave 3x3 for interpolation
end

function intensity = getIntensity(img, x, y)
if y < 1 || y > size(img, 1) || x < 1 || x > size(img, 2)
    intensity = 0;
else
    intensity = img(y, x, :);
end
end

function intensity = bilinear(img, pixel)
% Surrounding pixels
x1 = floor(pixel(1));
x2 = ceil(pixel(1));
y1 = floor(pixel(2));
y2 = ceil(pixel(2));

% Bilinear weights
w1 = pixel(1) - x1;
w2 = 1 - w1;
w3 = pixel(2) - y1;
w4 = 1 - w3;
w = [w4*w1; w4*w2; w3*w1; w3*w2];

% Surrounding pixel intensities
a1 = single(getIntensity(img, x1, y1));
a2 = single(getIntensity(img, x1, y2));
a3 = single(getIntensity(img, x2, y1));
a4 = single(getIntensity(img, x2, y2));

% Ignore surrounding black pixels
I1 = any(a1 >= 0);
I2 = any(a2 >= 0);
I3 = any(a3 >= 0);
I4 = any(a4 >= 0);
if ~(I1 || I2 || I3 || I4)
    intensity = uint8(0);
else
    intensity = w(1)*a1*I1 + w(2)*a2*I2 + w(3)*a3*I3 + w(4)*a4*I4;
    intensity = intensity / (w(1)*I1 + w(2)*I2 + w(3)*I3 + w(4)*I4);
    intensity = uint8(intensity);
end
end

function frame = stitch(frame, K, A)
% Get image dimensions
[Y, X, ~] = size(frame);

% Form resulting image
for i = 1:X
    for j = 1:Y
        % Compute location of original pixel
        pixel = A \ [i; j; 1];
        val1 = getIntensity(frame, i, j);
        val2 = bilinear(K, pixel);
%         val2 = getIntensity(K, i + 45, j);
        if all(val1 < 2) && any(val2 >= 2)
            frame(j, i, :) = val2;
        end
    end
end
end
