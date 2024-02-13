close all;
clear all;
clc;

% Agarro todas las imagenes sin modificar
imFolder = dir("rawGalaxyImgs\*.jpg");

% Cogemos cada imagen y la recortamos a 224x224, luego la guardamos en una
% nueva carpeta
for i = 1:50000
    filename = imFolder(i).name;
    im = imread("rawGalaxyImgs\"+filename);
    newIm = imcrop(im,[100, 100, 224, 224]);
    imwrite(newIm,"croppedGalaxyImgs\"+filename);
end
