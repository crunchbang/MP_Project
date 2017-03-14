I = imread('potholes.jpg');
G = rgb2gray(I);
corners = detectMinEigenFeatures(G);
imshow(G); hold on;
plot(corners.selectStrongest(500));
