clear all;
close all;
clc;

% Ejemplo red neuronal que emule puerta XOR a mano

% La red neuronal no consigue emular una puerta XOR, pues solo tiene dos
% parámetros y no es posible con tan pocos. Sin embargo, esto está hecho a
% propósito, pues así se puede visualizar la función de pérdida y el
% proceso de entrenamiento. 

% Creo mis vectores que tendrán el training set
num = 30000;
lim = 5; %(Límite de los pesos)
dataX = zeros(num,2);
dataY = zeros(num,1);

% Creamos los datos X e Y a mano. Para una puerta XOR tenemos dos input,
% que pueden ser 0 o 1, y un output, también 0 o 1. Solo si uno de los
% input es 1, el output será cero
for i = 1:num
    dataX(i,1) = randi(2,1)-1;
    dataX(i,2) = randi(2,1)-1;
    if dataX(i,1) || dataX(i,2)
        dataY(i) = 1;
    end
    if dataX(i,1) && dataX(i,2)
        dataY(i) = 0;
    end
end

% Creación de las capas, una de input con dos neuronas y otra de output con
% 1 neurona
lI = Layer(2, 1, "id");
lO = Layer(1, 2, "id");

% Declaramos la función de activación y la de pérdida (que es MSE)
sig = @(x)(1./(1+exp(-x)));
loss = @(x,y)(1./num .* sum(dataY-sig(x.*dataX(:,1)+y.*dataX(:,2))).^2);

% Creación del modelo xorNet con las dos capas
xorNet = Model([lI,lO]);

% Inicializamos los pesos (esto ya lo hace Model, pero de esta manera los
% resultados son mas diversos)
xorNet.Layers{2}.W = 2.*lim.*rand([1,2])-lim;

% Hacemos display del resultado de la red neuronal sin entrenar al procesar
% el primer ejemplo de entrenamiento y miramos los pesos y la pérdida
disp("Output de ["+dataX(1,1)+", "+dataX(1,2)+"] antes del entrenamiento: "+xorNet.forward(dataX(1,:)));
disp("Pesos antes del entrenamiento: ["+xorNet.Layers{2}.W(1)+", "+xorNet.Layers{2}.W(2)+"]");
disp("Loss: "+loss(xorNet.Layers{2}.W(1),xorNet.Layers{2}.W(2)));

% Creamos una visualización de la función de loss. En este caso, pondremos
% el bias a 0 para poder ver la función en 2D
xorNet.Layers{2}.B = 0;

N = 70;
w1 = linspace(-5,5,N);
w2 = linspace(-5,5,N);
[w1,w2] = meshgrid(w1,w2);

for i = 1:N
    for j = 1:N
        y(i,j) = loss(w1(i,j),w2(i,j));
    end
end
hold on;
surf(w1,w2,y);

% Ploteamos el primer punto con los pesos actuales
plot3(xorNet.Layers{2}.W(1), xorNet.Layers{2}.W(2), loss(xorNet.Layers{2}.W(1),xorNet.Layers{2}.W(2)),"rx","LineWidth",5,'MarkerSize',30);

% Calculamos la precisión actual de la red sin entrenar
succ = 0;
for i = 1:num
    res = round(xorNet.forward(dataX(i,:)));
    if res == dataY(i)
        succ = succ + 1;
    end
end
disp("Precisión antes: "+succ/num);
fprintf("\n");

% -
% Entrenamiento
% -

% Hiperparámetros
learningRate = 0.8;
miniBatch = 3000;

% Puntos de visualización
j = 2;
px(1) = xorNet.Layers{2}.W(1);
py(1) = xorNet.Layers{2}.W(2);
pz(1) = loss(xorNet.Layers{2}.W(1),xorNet.Layers{2}.W(2));

% Bucle de entrenamiento
for i = 1:length(dataX)/miniBatch    
    % Separamos una parte del training set para entrenar sobre ella
    X = dataX((i-1)*miniBatch+1:i*miniBatch,:);
    Y = dataY((i-1)*miniBatch+1:i*miniBatch,:);

    % Calculamos la activación (feed forward)
    A = sig(xorNet.Layers{2}.W*X');

    % Calculamos el gradiente de la función loss
    dW1 = 2./miniBatch .* sum( (Y - A').*X(:,1) );
    dW2 = 2./miniBatch .* sum( (Y - A').*X(:,2) );

    dW = [dW1,dW2];
                
    % Actualizamos los pesos
    xorNet.Layers{2}.W = xorNet.Layers{2}.W + learningRate .* dW;

    % Guardamos el paso intermedio de la función loss
    px(j) = xorNet.Layers{2}.W(1);
    py(j) = xorNet.Layers{2}.W(2);
    pz(j) = loss(xorNet.Layers{2}.W(1),xorNet.Layers{2}.W(2));
    j = j + 1;
end

% Dibujamos el paso intermedio en nuestra gráfica de la función loss
plot3(px,py,pz,"-o","Color","w","MarkerFaceColor","w");
xlabel("Peso 1");
ylabel("Peso 2");
zlabel("Loss");

% Miramos el output, los pesos de la red entrenada y la pérdida
disp("Output de ["+dataX(1,1)+", "+dataX(1,2)+"] después del entrenamiento: "+xorNet.forward(dataX(1,:)));
disp("Pesos después del entrenamiento: ["+xorNet.Layers{2}.W(1)+", "+xorNet.Layers{2}.W(2)+"]");
disp("Loss: "+loss(xorNet.Layers{2}.W(1),xorNet.Layers{2}.W(2)));

% Calculamos la precisión de la red entrenada
succ2 = 0;
for i = 1:num
    res = round(xorNet.forward(dataX(i,:)));
    if res == dataY(i)
        succ2 = succ2 + 1;
    end
end

disp("Precisión después: "+succ2/num);

% Ploteamos la pérdida final de la red entrenada
plot3(xorNet.Layers{2}.W(1),xorNet.Layers{2}.W(2),loss(xorNet.Layers{2}.W(1),xorNet.Layers{2}.W(2)),"gx","LineWidth",5,'MarkerSize',30);
hold off;

