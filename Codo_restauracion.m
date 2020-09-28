clc; close all; clear all;

imag = imread('codo_ruido.tif');
imag = rgb2gray(imag);
i = im2double(imag);

%%%Dimenciones de la imagen
[height,width] = size(i);

%%%Transformada de furier
f = fft2(i);
g = fftshift(f);

%%%Filtro notch
notch = [196 159; 228 59; 131 359; 97 458];
dist = [105 210 105 210];

[x,y] = meshgrid(1:width ,1:height);
filtroFFT = ones(size(x));

for n = 1:size(notch)
    
    filtroFFT = filtroFFT.*(1-exp(-((x-notch(n,1)).^2+(y-notch(n,2)).^2)/dist(n)^2));
    
end

fil_notch = g.*filtroFFT;
im_notch = ifftshift(fil_notch);
im_notch = real(ifft2(im_notch));

%%%Histograma ecualizado
equ = im_notch*5.5;

%%%Segmanetacion

%prewitt
hx = [1 1 1]'*[-1 0 1];

hy = hx';

%visualiza los datos
A = double(equ);

%gradiente en x
Gx = conv2(A,hx,'same');

%gradiente en y
Gy = conv2(A,hy,'same');

%magnitud del gradiente
G = sqrt(Gx.^2 + Gy.^2);
G = G*3;

%Umbral
u = 0.3;
GB = G >= u;

%%% area
area = bwarea(GB)

%%%Perimetro
perimetro = bwperim(GB)

%%%Compacidad
perimetro1 = 5000;
compacidad = (perimetro1*perimetro1)/(4*pi*area)

%%%Imagenes resultantes
figure;
imshow(imag); colormap gray;    title('Imagen Original');

% %%%Filtros
% figure();
% imshow(log(abs(g)),[0 10]);     title('FFT');
% 
% figure();
% imshow(filtroFFT,[0 1]);    title('filtro');
% 
% figure();
% imshow(log(abs(fil_notch)), [0 10]);   title('filtro2');

figure();
imshow(im_notch);  title('Filtro Notch');

figure();
imshow(equ);    title('Imagen Ecualizada');

figure; 
imshow(G); colormap gray;   title('Gradiente');

figure; 
imshow(GB); colormap gray;  title('Imagen Segmentada');

figure();
imshow(perimetro)  ;title('Perimetro');
