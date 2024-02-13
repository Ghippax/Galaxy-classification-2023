close all;
clear all;
clc;

filename1 = "MNIST_Data\train-labels.idx1-ubyte";
filename2 = "MNIST_Data\train-images.idx3-ubyte";
 
numTrain = 60000;

% Cargamos las imagenes y labels con una función externa que puede leer el
% formato del archivo
[imgs, labels] = readMNIST(filename2,filename1,numTrain,0);
dataY = zeros(numTrain, 10);
dataX = zeros(numTrain, 400);

% Transformamos el output de la función anterior en algo que pueda usar
% nuestra implementación de redes neuronales
for i = 1:numTrain
   vec = zeros(1,10);
   vec(labels(i)+1) = 1;
   dataY(i,:) = vec;
   dataX(i,:) = reshape(imgs(:,:,i), 1, []);
end

% Creamos una red con una capa oculta de 512 neuronas
lI = Layer(400, 1, "lRelu");
l1 = Layer(512, 400, "lRelu");
lO = Layer(10, 512, "softmax");

numNet = Model([lI,l1,lO]);

% Vemos como es el output de una imagen (esto es para comprobar que todo
% funciona como debería, además de dar un punto de comparación cuando
% volvamos hacer esto con la red ya entrenada)
disp("Resultado de pasar la primera imagen");
numNet.forward(dataX(1,:))
disp("Valor esperado");
dataY(1,:)

% Entrenamos la red, especificando los hiperparámetros
numNet.train(dataX,dataY, 0.0015, 32);

% Creamos una gráfica para visualizar la evolución de la pérdida con el
% tiempo
plot(1:length(numNet.trainLoss),numNet.trainLoss,"-o", "MarkerSize", 1, "MarkerEdgeColor", "red", "MarkerFaceColor", "red");
xlabel("Iteración");
ylabel("Loss");
title("Loss de la nn a lo largo del entrenamiento");

% Miramos ahora el resultado de pasar una imagen, e imprimimos la pérdida
% final
disp("Resultado de pasar la primera imagen")
numNet.forward(dataX(1,:))
disp("Valor esperado")
dataY(1,:)
disp("Loss final: "+round(numNet.loss(dataY,numNet.forward(dataX)),4,"significant"))

% Calculamos la precisión de la red
succ = 0;
for i = 1:numTrain
    vec = numNet.forward(dataX(i,:));
    [m,I] = max(vec);
    if I-1 == labels(i)
        succ = succ + 1;
    end
end
acc = succ./numTrain*100;
disp("Precisión en training set: "+round(acc,3,"significant")+"%");

% Por último, guardamos todo
save("MNIST_"+round(acc)+"acc.mat");
