close all;
clear all;
clc;

% Cargamos la red
load("Modelos\massiveGalaxyNet_29acc.mat");
net = trainedNet;

% Cargamos las imágenes y las clasificamos
numImgs = 10000;
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
testData = data(:,2:38);

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

labels = labels(1,1:numImgs);

% Vemos la matriz de confusión
confusionchart(labels, predLabels);
