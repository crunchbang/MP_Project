I = imread('potholes.jpg');
G = rgb2gray(I);
[featureVector,hogVisualization] = extractHOGFeatures(G);
figure;
imshow(G);
hold on;
plot(hogVisualization);