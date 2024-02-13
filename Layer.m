 classdef Layer < handle
    % Creamos el objeto capa, que contendra todas las neuronas de una capa
    % y algunas funciones que representan las operaciones que estas llevan
    % a cabo

    properties
        B % Matriz de sesgos
        W % Matriz de pesos
        fun % Función de activación (función anónima)
        f % Texto que dice cual es la función de activación
        a % Activaciones de la neurona
        z % Paso intermedio de las activaciones (antes de la función de activación)
    end

    methods
        % Constructor del objeto capa
        function obj = Layer(nNeurons, pastNeurons, f)
            % Tomo el numero de neuronas de esta capa, el de la anterior y
            % una función de activación
            
            % Creo la matriz de sesgos
            obj.B = randn(nNeurons,1);
            % Creo la matriz de pesos
            obj.W = randn(nNeurons,pastNeurons);
            
            obj.f = f;
            % Temperatura (sirve para mejorar la estabilidad numérica de la
            % función softmax
            T = 100;

            %Funciones de activación
            if(f == "sig")
                obj.fun = @(x)(1./(1+exp(-x)));
            elseif(f == "lRelu")
                obj.fun = @(x)(max(0.01.*x,x));
            elseif(f == "id")
                obj.fun = @(x)(x);
            elseif(f == "softmax") %La temperatura y el -max(x) es para aumentar la estabilidad numérica (evitar NaNs)
                obj.fun = @(x)(exp((x-max(x))./T)./sum(exp((x-max(x))./T),1));
            elseif(f == "lsoftmax")
                obj.fun = @(x)(log(exp(x./T)./sum(exp(x./T),1)));
            end
        end
        
        % Computa la derivada de la función de activación
        function output = zDer(obj, x)
            if obj.f == "sig"
                output = obj.fun(x);
                output = output.*(ones(size(output,1),size(output,2)) - output);
            elseif obj.f == "lRelu"
                output = x;
                output(output > 0) = 1;
                output(output <= 0) = 0.01.*output(output <= 0);
            elseif obj.f == "id"
                output = x;
            elseif obj.f == "softmax"
                output = obj.fun(x);
                output = output.*(ones(size(output,1),size(output,2)) - output);
            elseif obj.f == "lsoftmax"
                output = obj.fun(x);
                output = (ones(size(output,1),size(output,2)) - output);
            end
        end
        
        % Pasa un input por la capa, realizando todas las computaciones
        % pertinentes (funciona como un feedForward de una red neuronal de
        % 1 capa)
        function Y = eval(obj,X)
            % Primero realiza las operaciones lineales, multiplicando por
            % la matriz de pesos y sumando los sesgos, y luego lo pasa por
            % la función de activación, para obtener las activaciones
            % finales de cada neurona
            obj.z = obj.W*X+obj.B;
            obj.a = obj.fun(obj.z);
            Y = obj.a;
        end
    end
end