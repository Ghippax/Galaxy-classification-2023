close all;
clear all;
clc;

% Lee la tabla con los datos de las imagenes del training set
data = readtable("galaxyData.csv");

% Coge de la tabla la informacion de la categoria
data = data{1:end,1:end};
testData = data(:,2:4);

% Crea un vector "labels" con la clasificación de cada galaxia siguiendo la
% información de la tabla
for i = 1:length(testData)
    if max(testData(i,:)) == testData(i,1)
        labels(i) = "Elíptica";
    else
        labels(i) = "Espiral";
    end
end

% Convierte a "labels" en un vector de categorias y limita su tamaño a
% 50000 imagenes
labels = categorical(labels);
labels = labels(1:50000);

% Pasa las imagenes del training set y el vector de labels a la variable imds, que es la aceptada por la función trainNetwork 
imds = imageDatastore("croppedGalaxyImgs\","Labels",labels);

% Dividimos las imagenes 90-10, 90% para entrenar, 10% para estudiar la
% precisión de la red
[imdsTrain, imdsValidate] = splitEachLabel(imds, 0.9, "randomized");
imdsTrain2 = augmentedImageDatastore([224 224 3],imdsTrain);
imdsValidate2 = augmentedImageDatastore([224 224 3],imdsValidate);

% Tamaño de las imagenes y numero de categorias
inputSize = [224 224 3];
numClasses = 2;

% Cargamos el modelo resnet50, ya preentrenado para clasificar imagenes de
% forma general, y lo pasamos a una forma usable por las funciones que nos
% interesan
net = resnet50;
lgraph = layerGraph(net);

% Reemplazamos una capa de las resnet por una fullyConnected que permite
% que clasifique en dos las imagenes
newLearnableLayer = fullyConnectedLayer(numClasses, ...
        Name="new_fc", ...
        WeightLearnRateFactor=10, ...
        BiasLearnRateFactor=10);
    
lgraph = replaceLayer(lgraph,"fc1000",newLearnableLayer);

% Reemplazamos la última capa, para que nos dé el resutado correcto
newOutputLayer = classificationLayer("Classes",["Elíptica","Espiral"]);
lgraph = replaceLayer(lgraph,"ClassificationLayer_fc1000",newOutputLayer);

% Especificamos los hiperparámetros de la red, poniendo especial atención
% al learningRate, miniBacthSize y MaxEpochs
options = trainingOptions("sgdm", ...
    InitialLearnRate=0.0005, ...
    MiniBatchSize=32, ...
    MaxEpochs=10, ...
    Verbose= false, ...
    ValidationData=imdsValidate2, ...
    ValidationFrequency=100, ...
    ValidationPatience=5, ...
    Plots="training-progress");

% Finalmente, iniciamos el entrenamiento de la red
trainedNet = trainNetwork(imdsTrain2,lgraph,options);

% Guardamos los resultados
save("matlabGalaxyNet");
