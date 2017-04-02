I = imread('potholes.jpg');
G = rgb2gray(I);
corners = detectFASTFeatures(G);
imshow(G); hold on;
plot(corners.selectStrongest(1500));
imwrite(G, 'cornersRoadsFast.jpg');