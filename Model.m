classdef Model < handle
    % Crea una red neuronal (conjunto de capas)

    properties
        Layers % Array de células, cada una con una capa
        trainLoss % Vector con la pérdida a lo largo del entrenamiento
    end

    methods
        % Función constructora de redes neuronales
        function obj = Model(L)
            % Tomo un array de capas y lo paso a mi array celular
            obj.Layers = cell(1,length(L));
            obj.trainLoss = [];
            for i = 1:length(L)
                obj.Layers{i} = L(i);
            end
        end
        
        % Toma un input y lo pasa, capa por capa, por la red neuronal,
        % hasta obtener el output
        function Y = forward(obj,X)
            % Cojo la transpuesta de mi input (por convención propia, el
            % input tiene los ejemplos en cada fila, y el contenido de cada
            % ejemplo en las columnas. Sin embargo, los cálculos los hago
            % tomando este orden al revés, por ello hago la traspuesta)
            Y1 = X';
            % Paso este input por cada capa, haciendo los cálculos con la
            % función "eval"
            for i = 2:length(obj.Layers)
                Y2 = obj.Layers{i}.eval(Y1);
                Y1 = Y2;
            end
            Y = Y1;
        end

        % Calcular el gradiente (de la función de coste) en función de los
        % parametros de cada capa de la red
        function [dW, dB] = backprop(obj, testX, testY)
            %TestX es una porción del training set
            %TestY es el output esperado para cada ejemplo del training set

            %dW es el gradiente de los pesos, del tipo cell
            %dB es el gradiente de los bias, del tipo cell
            %L el número de capas
            L = length(obj.Layers);
            dW = cell(1, L-1);
            dB = cell(1, L-1);
            % Los z (pasos intermedios) y las activaciones de cada capa
            zs = cell(1, L-1);
            acts = cell(1, L-1);
            
            % La primera activación corresponde a testX (traspuesto, por
            % los motivos explicados en forward)
            acts{1} = testX';

            % Realizo un feed forward de todos los ejemplos de testX a la
            % vez, guardando z y activaciones de capa capa
            for i = 2:L
                zs{i} = obj.Layers{i}.W * acts{i-1} + obj.Layers{i}.B;
                acts{i} = obj.Layers{i}.fun(zs{i});
            end
            
            % output predicho
            Ypred = acts{L}; 

            % Paso hacia atras ("backward pass"), calculo el gradiente.
            % Calculo delta, que es comun a dW y dB
            delta{L} = -(testY'./Ypred).* obj.Layers{i}.zDer(zs{L});
            for i = L-1:-1:2
                delta{i} = (obj.Layers{i+1}.W' * delta{i+1}) .* obj.Layers{i}.zDer(zs{i});
            end
            
            % Calculo con delta el gradiente en función de W y de B
            m = size(testY,1); % Numero de ejemplos en mi porció de training set
            for i = 1:L-1
                dW{i} = 1/m .* delta{i+1} * acts{i}';
                dB{i} = 1/m .* sum(delta{i+1}, 2);
            end
        end
    
        % Entrena la red neuronal, calculando los gradientes a cada paso
        % del entrenamiento y aplicándolos a la red neuronal
        function train(obj, X, Y, learningRate, perBatch)
            % X, Y, son el input y output del training set
            % learningRate y perBatch son mis hiperparámetros, learningRate
            % especifica lo mucho que cambian los parámetros a cada paso, y
            % perBatch cuantos ejemplos del traningSet tomo para cada paso

            % Numero de iteraciones (o pasos) del entrenamiento
            nIt = floor(length(X)./perBatch);
            for i = 1:nIt
                % Tomo la porción del trainingSet para el paso
                % correspondiente
                dataX = X((i-1)*perBatch+1:i*perBatch,:);
                dataY = Y((i-1)*perBatch+1:i*perBatch,:);

                % Calculo los gradientes
                [dW,dB] = obj.backprop(dataX,dataY);
                
                % Aplico el gradiente capa por capa
                for j = 2:length(obj.Layers)
                    obj.Layers{j}.W = obj.Layers{j}.W - learningRate * dW{j-1};
                    obj.Layers{j}.B = obj.Layers{j}.B - learningRate * dB{j-1};
                end

                % Calculo la perdida en este paso del entrenamiento, la
                % guardo en un vector y la muestro por consola
                obj.trainLoss(i) = obj.loss(Y,obj.forward(X));
                disp("Loss: "+round(obj.trainLoss(i),3,"significant")+" | "+round(100.*i./nIt,3,"significant") + "%");
            end
        end

        % Función de pérdida o coste de la red neuronal (Categorical Cross
        % Entropy)
        function l = loss(obj, Y, tY)
            l = 1/length(Y) .* sum(-sum(Y.*log(tY + 1e-50)', 1));
        end
 
    end
end