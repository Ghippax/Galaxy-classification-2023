close all;
clear all;
clc;

% Cargamos las imágenes y respuestas y las preparamos para ser usadas
filename1 = "MNIST_Data\train-labels.idx1-ubyte";
filename2 = "MNIST_Data\train-images.idx3-ubyte";
 
numTrain = 60000;

[imgs, labels] = readMNIST(filename2,filename1,numTrain,0);
dataY = zeros(numTrain, 10);
dataX = zeros(numTrain, 400);

for i = 1:numTrain
   vec = zeros(1,10);
   vec(labels(i)+1) = 1;
   dataY(i,:) = vec;
   
   dataX(i,:) = reshape(imgs(:,:,i), 1, []);
end

% Cargamos el modelo e inicializamos la matriz de confusión
load("Modelos\MNIST_74acc.mat");

accMatrix = zeros(10,10);

% Calculamos la matriz de confusión
for i = 1:length(dataX)
    [el,j] = max(dataY(i,:));
    [el,k] = max(numNet.forward(dataX(i,:)));
    accMatrix(j,k) = accMatrix(j,k) + 1;
end

% Visualizamos la matriz
confusionchart(accMatrix);
