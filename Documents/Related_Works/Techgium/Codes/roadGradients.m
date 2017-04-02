I = imread('smooth.jpg');
G = rgb2gray(I);
[Gmag, Gdir] = imgradient(G,'prewitt');

figure; imshowpair(Gmag, Gdir, 'montage');
title('Gradient Magnitude, Gmag (left), and Gradient Direction, Gdir (right), using Prewitt method')
axis off;
%imwrite(Gmag, 'Gradient Magnitude.png');
BW1 = edge(G,'Sobel');
BW2 = edge(G,'Prewitt');
imshow(BW1);
imshow(BW2);
%imwrite(BW1, 'Sobel.png');
%imwrite(Gmag, 'Prewitt.png');

[Gx, Gy] = imgradientxy(G);
[Gmag, Gdir] = imgradient(Gx, Gy);
imwrite(Gmag,'Gmag.jpg');
figure, imshow(Gmag, []), title('Gradient magnitude')
figure, imshow(Gdir, []), title('Gradient direction')
title('Gradient Magnitude and Gradient Direction')
figure; imshowpair(Gx, Gy, 'montage'); axis off;
imwrite(Gy, 'Gy.jpg');
title('Directional Gradients, Gx and Gy')
%imwrite(Gmag, 'Gradient Magnitude.jpg');
%imwrite(Gdir, 'Gradient Direction.jpg');