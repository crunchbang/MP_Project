I = imread('smooth.jpg');
G = rgb2gray(I);

BW2 = edge(G,'Sobel');
BW3 = edge(G, 'Canny');
imshow(BW1);
imshow(BW2);
imwrite(BW3, 'smoothCanny.png');
imwrite(BW2, 'smoothSobel.png');