close all;
clear all;
clc;

% Cargamos la red
load("Modelos\simpleGalaxyNet_87acc.mat");
net = trainedNet;

% Cargamos las imágenes y las clasificamos
numImgs = 20000;
imFolder = dir("croppedGalaxyImgs\*.jpg");

for i = 1:numImgs
    filename = imFolder(i).name;
    im = imread("croppedGalaxyImgs\"+filename);
    im = imresize(im, [224,224]);
    predLabels(i) = classify(net,im);
    disp("Cargando imágenes: "+round(i./numImgs.*100,3,"significant")+"% ("+i+" de "+numImgs+")");
end

% Calculamos los labels de verdad
data = readtable("galaxyData.csv");

data = data{1:end,1:end};
testData = data(:,2:4);

for i = 1:length(testData)
    if max(testData(i,:)) == testData(i,1)
        labels(i) = "eliptica";
    else
        labels(i) = "espiral";
    end
end

labels = labels(1,1:numImgs);

% Vemos la matriz de confusión
confusionchart(labels, predLabels);
