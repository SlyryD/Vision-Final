% Read in images
idxs = 81:376;
n = length(idxs);
imgs = cell(n, 1);
for i = 1:n
    imgs{i} = imread(sprintf('.\\images\\rgb\\frame_%d_rgb.png', idxs(i)));
end

% % Compute optical flows
% opticalFlow = vision.OpticalFlow('OutputValue', 'Horizontal and vertical components in complex form');
% flows = cell(n, 1);
% for i = 1:n
%     flows{i} = step(opticalFlow, single(rgb2gray(imgs{i})));
% end

% Compute optical flows
flows = cell(n - 1, 1);
for i = 1:(n - 1)
    flows{i} = compute_flow(imgs{i}, imgs{i + 1});
end
