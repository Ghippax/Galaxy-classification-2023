close all;
clear all;
clc;

% Leo la tabla con los datos de cada imagen
data = readtable("galaxyData.csv");

% Convertimos el tipo cell a matriz y cogemos las columnas pertinentes
data = data{1:end,1:end};
testData = data(:,2:38);

% Creamos todas las labels del dataset, encadenando las distintas
% posibilidades que encontramos en el dataset
for i = 1:length(testData)
    labels(i) = "_";
    if testData(i,3) >= testData(i,1) || testData(i,3) >= testData(i,2)
        labels(i) = labels(i) + "artifact_"; % Artefacto/estrella
    elseif testData(i,2) >= testData(i,1)
        labels(i) = labels(i) + "disk_"; % Espiral
    else
        labels(i) = labels(i) + "smooth_"; % Suave
    end

    if testData(i,4) >= testData(i,5)
        labels(i) = labels(i) + "edgeOn_"; % Disco visto de lado
    end

    if testData(i,6) >= testData(i,7)
        labels(i) = labels(i) + "bar_"; % Barra en el centro
    end

    if testData(i,8) >= testData(i,9)
        labels(i) = labels(i) + "espiral_"; % Patron espiral
    end

    if testData(i,10) >= max(testData(i,10:13))
        labels(i) = labels(i) + "noBulge_"; % No bulto en el centro
    elseif testData(i,11) >= max(testData(i,10:13))
        labels(i) = labels(i) + "smBulge_"; % Bulto pequeño
    elseif testData(i,12) >= max(testData(i,10:13))
        labels(i) = labels(i) + "bigBulge_"; % Bulto grande
    else
        labels(i) = labels(i) + "domBulge_"; % Bulto dominante
    end

    if testData(i,14) >= testData(i,15)
        labels(i) = labels(i) + "odd_"; % Rara
        if testData(i,19) >= max(testData(i,19:25))
            labels(i) = labels(i) + "ring_"; % Anillo
        elseif testData(i,20) >= max(testData(i,19:25))
            labels(i) = labels(i) + "lens_"; % Lente o arco
        elseif testData(i,21) >= max(testData(i,19:25))
            labels(i) = labels(i) + "disturbed_"; % Disturbed
        elseif testData(i,22) >= max(testData(i,19:25))
            labels(i) = labels(i) + "irregular_"; % Irregular
        elseif testData(i,23) >= max(testData(i,19:25))
            labels(i) = labels(i) + "other_"; % Misc
        elseif testData(i,24) >= max(testData(i,19:25))
            labels(i) = labels(i) + "merger_"; % Fusion
        else
            labels(i) = labels(i) + "dustLane_"; % Polvo
        end
    end

    if testData(i,16) >= testData(i,17) || testData(i,16) >= testData(i,18)
        labels(i) = labels(i) + "round_"; % Redonda
    elseif testData(i,17) >= testData(i,18)
        labels(i) = labels(i) + "semiRound_"; % Poco redonda
    else
        labels(i) = labels(i) + "cigar_"; % Muy ovalada
    end
    
    if testData(i,26) >= testData(i,27) || testData(i,26) >= testData(i,28)
        labels(i) = labels(i) + "rounded_"; % Bulto redondo
    elseif testData(i,27) >= testData(i,28)
        labels(i) = labels(i) + "boxy_"; % Bulto cuadrado
    end

    if testData(i,29) >= testData(i,30) || testData(i,29) >= testData(i,31)
        labels(i) = labels(i) + "tight_"; % Apretado
    elseif testData(i,30) >= testData(i,31)
        labels(i) = labels(i) + "medium_"; % Mediano
    else
        labels(i) = labels(i) + "loose_"; % Suelto
    end

    if testData(i,32) >= max(testData(i,32:37))
        labels(i) = labels(i) + "1ring_"; % 1 anillo
    elseif testData(i,33) >= max(testData(i,32:37))
        labels(i) = labels(i) + "2ring_"; % 2 anillos
    elseif testData(i,34) >= max(testData(i,32:37))
        labels(i) = labels(i) + "3ring_"; % 3 anillos
    elseif testData(i,35) >= max(testData(i,32:37))
        labels(i) = labels(i) + "4ring_"; % 4 anillos
    elseif testData(i,36) >= max(testData(i,32:37))
        labels(i) = labels(i) + ">4_"; % Mas de 4
    else
        labels(i) = labels(i) + "dontk_"; % No sabe
    end
end

% Convertimos este vector en uno categórico y lo limitamos a 50000
% elementos
labels = categorical(labels);
labels = labels(1:50000);

% Pasamos a formato imds y dividimos las imagenes en training set y
% validation set
imds = imageDatastore("croppedGalaxyImgs\","Labels",labels);
[imdsTrain, imdsValidate] = splitEachLabel(imds, 0.9, "randomized");
imdsTrain2 = augmentedImageDatastore([224 224 3],imdsTrain);
imdsValidate2 = augmentedImageDatastore([224 224 3],imdsValidate);

% Escogemos el tamaño de las imagenes y el número de categorias
inputSize = [224 224 3];
numClasses = 3166;

% Cargamos una reed preentrenada
net = resnet50;
lgraph = layerGraph(net);

% Sustituimos las últimas capas para convertirlo en una red clasificadora
newLearnableLayer = fullyConnectedLayer(numClasses, ...
        Name="new_fc", ...
        WeightLearnRateFactor=10, ...
        BiasLearnRateFactor=10);
    
lgraph = replaceLayer(lgraph,"fc1000",newLearnableLayer);

newOutputLayer = classificationLayer("Classes",categories(labels));
lgraph = replaceLayer(lgraph,"ClassificationLayer_fc1000",newOutputLayer);

% Especificamos los hiperparámetros
options = trainingOptions("sgdm", ...
    InitialLearnRate=0.0005, ...
    MiniBatchSize=32, ...
    MaxEpochs=2, ...
    Verbose= false, ...
    ValidationData=imdsValidate2, ...
    ValidationFrequency=100, ...
    ValidationPatience=5, ...
    Plots="training-progress");

% Entrenamos la red
trainedNet = trainNetwork(imdsTrain2,lgraph,options);

% Guardamos los resultados
save("massiveGalaxyNet");
