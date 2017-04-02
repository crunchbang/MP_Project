I = imread('potholes.jpg');
corners = detectHarrisFeatures(I);
imshow(I); hold on;
plot(corners.selectStrongest(500));