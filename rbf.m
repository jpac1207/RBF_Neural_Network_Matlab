% ---------- Parâmetros Gerais ----------
maxEpochs = 10; % Número de épocas do treinamento
numberOfTrainings = 10; % Número de treinamentos a serem utilizados
H = 7; % Número de neurônios na camada escondida
I = 6; % Número de neurônios na camada de entrada
O = 4; % Número de neurônios na camada de saída
eta = 0.05; % Learning Rate utilizado no cálculo do backpropagation.
eta_gaussian = 0.05; % Learning Rate utilizado no cálculo da atualização de centro dos neurônios de ativação gaussiana.

% ---------- Mapas a serem utilizados no pré processamento de dados ----------
preProcessingConfig.buyingMap = containers.Map({'vhigh', 'high', 'med', 'low'}, {5, 4, 3, 2});
preProcessingConfig.maintMap = containers.Map({'vhigh', 'high', 'med', 'low'}, {5, 4, 3, 2});
preProcessingConfig.doorsMap = containers.Map({'2', '3', '4', '5more'}, {2, 3, 4, 5});
preProcessingConfig.personsMap = containers.Map({'2', '4', 'more'}, {2, 4, 5});
preProcessingConfig.lugBootMap = containers.Map({'small', 'med', 'big'}, {1, 2, 3});
preProcessingConfig.safetyMap = containers.Map({'low', 'med', 'high'}, {1, 2, 3});
preProcessingConfig.labelMap = containers.Map({'unacc', 'acc', 'good', 'vgood'}, {1, 2, 3, 4});

%testRow = 1212;
%predictExampleUsingBestWeights(preProcessingConfig, activationType, testRow);

% ---------- Chamadas de funções para computação de métricas ----------

% Realiza treinamento da RBF 'numberOfTrainings' vezes.
%doTraining(preProcessingConfig, maxEpochs, numberOfTrainings, I, H, O, eta, eta_gaussian);

% Realiza treinamento da MLP 'numberOfTrainings' vezes variando o número de neurônios da camada escondida.
%doTrainingWithHiddenLayerSizeVariation(preProcessingConfig, maxEpochs, numberOfTrainings, I, 5, 15, O, eta, activationType);

% Realiza treinamento da MLP 'numberOfTrainings' vezes variando a taxa de aprendizado.
%doTrainingWithEtaVariation(preProcessingConfig, maxEpochs, numberOfTrainings, I, H, O, [0.05 0.01 0.05 0.1 0.15], eta_gaussian)   

% ---------- Implementações das funções de computação de métricas ----------

% Realiza 'numberOfTrainings' treinamentos, obtendo ao final:
% Melhor erro de treinamento encontrado
% Média dos erros de treinamento
% Média dos erros de validação
% Gráfico com os erros médios por epóca
function doTraining(preProcessingConfig, maxEpochs, numberOfTrainings, I, H, O, eta, eta_gaussian)
    data = readData('./data/car.data');
    [X, Y] = preProcessing(data, preProcessingConfig);
    X_norm = normalizeInput(X);
    [X_train, Y_train, X_val, Y_val, X_test, Y_test] = splitData(X_norm, Y);
    finalErrors = zeros(maxEpochs, 1);  
    finalValErrors = zeros(maxEpochs, 1);
    bestError = 1;     
    
    for i = 1:numberOfTrainings
        [hiddenVsInputWeights, outputVsHiddenWeights, outputVsHiddenBias, errors, valErrors]  = trainRBF(I, H, O, maxEpochs, eta, eta_gaussian, ...
            X_train', Y_train, X_val', Y_val); 
        finalErrors = finalErrors + errors;
        finalValErrors = finalValErrors + valErrors;
        if(errors(maxEpochs) < bestError)
            bestError = errors(maxEpochs);
            save('bestWeights.mat', 'hiddenVsInputWeights', 'outputVsHiddenWeights', 'outputVsHiddenBias');
        end        
    end
    meanFinalErrors = (finalErrors./numberOfTrainings);
    meanFinalValErrors = (finalValErrors./numberOfTrainings);
    bestError
    meanFinalError = meanFinalErrors(maxEpochs)
    meanFinalValError = meanFinalValErrors(maxEpochs)
    plot((1:maxEpochs), meanFinalErrors, 'o');
    hold on;
    plot((1:maxEpochs), meanFinalValErrors, 'x');
    hold off;
    legend('Média Erros Treinamento', 'Média Erros Validação');
end

% Realiza 'numberOfTrainings' treinamentos, variando a quantidade de neurônios da camada escondida ['H_init', 'H_end']. Obtendo ao final:
% Melhor erro de treinamento encontrado
% Média dos erros de treinamento
% Média dos erros de validação
% Gráfico com os erros médios por epóca
function doTrainingWithHiddenLayerSizeVariation(preProcessingConfig, maxEpochs, numberOfTrainings, I, H_init, H_end, O, eta, eta_gaussian)
   H = H_init;
   while H <= H_end
    H
    doTraining(preProcessingConfig, maxEpochs, numberOfTrainings, I, H, O, eta, eta_gaussian);
    H = H+1;
    pause;
   end
end

% Realiza 'numberOfTrainings' treinamentos, variando a taxa de aprendizado em função dos elementos do vetor 'etas'. Obtendo ao final:
% Melhor erro de treinamento encontrado
% Média dos erros de treinamento
% Média dos erros de validação
% Gráfico com os erros médios por epóca
function doTrainingWithEtaVariation(preProcessingConfig, maxEpochs, numberOfTrainings, I, H, O, etas, eta_gaussian)   
   i = 1;  
   while i <= size(etas, 2)
       etas(i)
       doTraining(preProcessingConfig, maxEpochs, numberOfTrainings, I, H, O, etas(i), eta_gaussian);
       i = i+1;
       pause;
   end
end

% Realiza predição do exemplo da linha 'rowOfExample' do dataset,
% utilizando os pesos salvos no arquivo 'bestWeights.mat', que deve se
% encontrar no mesmo diretório do arquivo aqui executado
function predictExampleUsingBestWeights(preProcessingConfig, activationType, rowOfExample)
    weightsStruct = load('bestWeights.mat');
    hiddenVsInputWeights = weightsStruct.hiddenVsInputWeights;
    hiddenVsInputBias = weightsStruct.hiddenVsInputBias;
    outputVsHiddenWeights = weightsStruct.outputVsHiddenWeights;
    outputVsHiddenBias = weightsStruct.outputVsHiddenBias;
    data = readData('./data/car.data');
    [X, Y] = preProcessing(data, preProcessingConfig);    
    prediction = testMLP(hiddenVsInputWeights, hiddenVsInputBias, outputVsHiddenWeights, outputVsHiddenBias, activationType, X(rowOfExample, :)');
    [~, real] = max(Y(:, rowOfExample));
    sprintf("Predição: %d", prediction)
    sprintf("Real: %d", real)
end

% Realiza predição da classe de um dado padrão de entrada 'X', utilizando
% os parâmetros: 
% hiddenVsInputWeights -> Matriz que representa os pesos aprendidos para as
% conexões entre 
function Y = testMLP(hiddenVsInputWeights, hiddenVsInputBias, outputVsHiddenWeights, outputVsHiddenBias, activationType, X)          
    net_h = hiddenVsInputWeights * X + hiddenVsInputBias * ones(1, size(X, 2));
    Yh = activation(activationType, net_h);
    net_o = outputVsHiddenWeights * Yh + outputVsHiddenBias * ones(1, size (Yh, 2));
    Y_net = exp(net_o)./sum(exp(net_o));
    [value, index] = max(Y_net);
    Y = index;
end

% Realiza o treinamento da RBF, de acordo com os parametros:
% I -> Número de neurônios na camada de entrada
% H -> Número de neurônios na camada escondida
% O -> Número de neurônios na camada de saída
% maxEpochs -> Número de epócas do treinamento
% eta -> Taxa de aprendizado
% activationType -> Flag utilizada para definir a função de ativação da
% camada escondida
% X_train -> Padrões de entrada utilizados durante o treinamento
% Y_train -> Padrões de saída utilizados durante o treinamento
% X_val -> Padrões de entrada utilizados na validação
% Y_val -> Padrões de saída utilizados na validação
function [hiddenVsInputWeights, outputVsHiddenWeights, outputVsHiddenBias, finalErrors, finalValErrors] = trainRBF(I, H, O, maxEpochs, eta, ...
    eta_gaussian, X_train, Y_train, X_val, Y_val)
    currentEpoch = 1;    
    errors = zeros(maxEpochs, 1);  
    validationErrors = zeros(maxEpochs, 1);
    % Número de padrões de entrada
    numberOfTrainingInstances = size(X_train, 2);
    % Número de padrões de validação
    numberOfValidationInstances = size(X_val, 2);
    % Número de atributos de entrada
    numberOfAttributes = size(X_train, 1);
    % Centros camada escondida
    C = rand(H, I) - 0.5;     
    % Pesos entre camada escondida e camada de saída
    Woh = rand (O, size(C, 2)) - 0.5;
    % Bias entre camada escondida e camada de saída
    bias_oh = rand(O, 1) - 0.5;    

    % ---------------------- Aplicação do Algoritmo WTA ----------------------     
    [nearestHiddenNeurons, C] = wta(X_train, C, eta_gaussian);
    
    % ---------------------- Determinação da abertura dos neurônios escondidos ----------------------

    % Considera os N/2 neurônios mais próximos para cálculo da abertura de
    % cada neurônio
    T = floor(H/2);
    % Vetor que irá armazenar a abertura para cada neurônio da camada escôndida
    sigma = zeros(H, 1);
    % Percorre todos os neurônios da camada escondida
    distancesBetweenHiddenNeurons = zeros(H, H) + realmax;    
    % Computa a distância entre cada par de neurônios 
    for i=1:H
        for j=i+1:H            
            distanceBetweenNeuronsIandJ = (sum((C(i, :) - C(j, :)).^2)/size(C,1));            
            distancesBetweenHiddenNeurons(i, j) = distanceBetweenNeuronsIandJ;
            distancesBetweenHiddenNeurons(j, i) = distanceBetweenNeuronsIandJ;
        end
    end
    
    % Computa a abertura de cada neurônio escondido     
    % Percorre todos os neurônios da camada escondida
    for i=1:H
        % Vetor que irá armazenar as T menores distâncias do neurônio i em
        % relação aos outros neurônios escondidos
        minDistances = zeros(T, 1);        
        for j=1:T                        
            [minValue, minPosition] = min(distancesBetweenHiddenNeurons(i, :));
            minDistances(j) = minValue;
            distancesBetweenHiddenNeurons(i, minPosition) = realmax;            
        end
        sigma(i) = sum(minDistances)/T;
    end    
     
    % ---------------------- Treinamento da camada de saída ----------------------
    error = 0;
    validationError = 0;
     while currentEpoch <= maxEpochs

        for i=1:numberOfTrainingInstances          
             % ------- Hidden Layer -------                   
             nearestHiddenNeuronPosition = nearestHiddenNeurons(i);
             nearestHiddenNeuron = C(nearestHiddenNeuronPosition, :);    
             nearestSigma = sigma(nearestHiddenNeuronPosition);
             mi_h = sqrt((X_train(:, i) - nearestHiddenNeuron').^2);
             out_h = exp(-((mi_h.^2)./(2*nearestSigma.^2)));             
            
             % ------- Output Layer -------              
             net_o = Woh * out_h + bias_oh * ones(1, size(out_h, 2));
             Y_net = exp(net_o)/sum(exp(net_o));   % Aplicação da softmax                 
             E = (-1).*sum((Y_train(:, i).*log(Y_net)));  % Computação do erro                   
             %sprintf("%f", E)                 

             % backward                 
             df =  (Y_train(:, i)-Y_net);
             delta_bias_oh = eta * sum((E.*df)')';             
             delta_Woh = eta * (E.*df)*out_h';
             Eh = (Woh')*(E.*df);
            
             %update weights  
             Woh = Woh + delta_Woh;
             bias_oh = bias_oh + delta_bias_oh;
             
             %calculate error                        
             error = error + sum(((Y_train(:, i) .* (1-Y_net)).^2), 'all');
        end        
        error = error/numberOfTrainingInstances;
        sprintf("%f", error);
        errors(currentEpoch) = error;

        for i=1:numberOfValidationInstances
             % ------- Hidden Layer -------                   
             nearestHiddenNeuronPosition = getNearestNeuronPosition(X_val(:, i), C);
             nearestHiddenNeuron = C(nearestHiddenNeuronPosition, :);    
             nearestSigma = sigma(nearestHiddenNeuronPosition);
             mi_h = sqrt((X_val(:, i) - nearestHiddenNeuron').^2);
             out_h = exp(-((mi_h.^2)./(2*nearestSigma.^2)));     
            
             % ------- Output Layer -------              
             net_o = Woh * out_h + bias_oh * ones(1, size(out_h, 2));
             val_Y_net = exp(net_o)/sum(exp(net_o));   % Aplicação da softmax  
             validationError = sum(((Y_val .* (1-val_Y_net))), 'all'); 
        end
        validationError = validationError/numberOfValidationInstances;
        sprintf("%f", error);
        validationErrors(currentEpoch) = validationError;

        currentEpoch = currentEpoch + 1;
   end     

    finalErrors = errors;
    finalValErrors = validationErrors;
    hiddenVsInputWeights = C;   
    outputVsHiddenWeights = Woh;
    outputVsHiddenBias = bias_oh;
end

function[minPosition] = getNearestNeuronPosition(inputPattern, hiddenNeurons)
    differences = zeros(size(hiddenNeurons, 1), 1);
    for j = 1:size(hiddenNeurons, 1)           
        absoluteDifference  = sum((inputPattern - hiddenNeurons(j, :)').^2);
        differences(j) = absoluteDifference;
    end       
    [~, minPosition] = min(differences);     
end

% Aplicação do algoritmo WTA. Recebe como argumento, a matrix 'inputMatrix'
% com os padrões de entrada, a matrix 'hiddenNeurons' com os pesos dos
% neurônios escondidos e o eta relativo a atualização dos centros dos neurônios escondidos. Como retorno, são devolvidos:
% 'nearestHiddenNeurons' -> Vetor coluna contendo a posição do neurônio
% mais próximo para cada padrão de entrada;
% 'hiddenNeurons' -> Neurônios escondidos com os valores de centro
% atualizados
function[nearestHiddenNeurons, hiddenNeurons] = wta(inputMatrix, hiddenNeurons, eta_gaussian)
    numberOfInstances = size(inputMatrix, 2);
    previousQuantizationError = realmax;
    howManyIterations = 0;
    maxOfIterations = 500;
    nearestHiddenNeurons = zeros(numberOfInstances, 1);
    while true       
        quantizationError = 0;
        % Percorre todos os vetores de entrada x
        for i = 1:numberOfInstances           
           % Para cada vetor de entrada, determina o centro mais próximo
            minPosition = getNearestNeuronPosition(inputMatrix(:, i), hiddenNeurons);
            % Atualiza o centro mais próximo
           hiddenNeurons(minPosition, :) = hiddenNeurons(minPosition, :) + eta_gaussian * (inputMatrix(:, i)' - hiddenNeurons(minPosition, :));
           nearestHiddenNeurons(i)  = minPosition;
           % Computa erro de quantização
           quantizationError = (quantizationError + sum((inputMatrix(:, i)' - hiddenNeurons(minPosition, :)).^2))/numberOfInstances;
        end       
        howManyIterations = howManyIterations + 1;        
        % Condições de Parada: maxOfIterations ou erro não diminuiu desde a
        % última iteração                
        if((quantizationError < previousQuantizationError) || howManyIterations >= maxOfIterations)
            previousQuantizationError = quantizationError;
        else            
            break;
        end    
    end  
end

% Normaliza os dados de entrada para [0,1]
function [X_output] =  normalizeInput(X_input)
    X_output = X_input;
    numberOfColumns = size(X_input, 2);
    % Para cada coluna
    for i = 1:numberOfColumns        
        X_max = max(X_output(:, i));
        X_min = min(X_output(:, i));   
        numerator = X_output(:, i) - X_min;
        denominator = (X_max-X_min);
        if denominator ~= 0
            X_output(:, i) = numerator./denominator;
        else
            X_output(:, i) = 0;
        end       
    end    
end

% Realiza o carregamento dos dados contidos no arquivo existente no caminho
% 'dataPath'
function data = readData(dataPath)
    data = importdata(dataPath, ',');
end

% Realiza a divisão dos dados contidos em 'X' e 'Y' em:
% X_train -> Padrões de entrada a serem utilizados no treino (70%)
% Y_train -> Padrões de saída a serem utilizados no treino (70%)
% X_val -> Padrões de entrada a serem utilizados na validação (20%)
% Y_val -> Padrões de saída a serem utilizados na validação (20%)
% X_test -> Padrões de entrada a serem utilizados no teste (10%)
% Y_test -> Padrões de saída a serem utilizados no testw (10%)
function [X_train, Y_train, X_val, Y_val, X_test, Y_test] = splitData(X, Y)
    numberOfRows = size(X, 1);
    trainProportion = 0.7;
    trainRows = floor(numberOfRows * trainProportion);
    valProportion = 0.2;
    valRows = floor(numberOfRows * valProportion);
    testProportion = 0.1;
    testRows = floor(numberOfRows * testProportion);

    randIndexes = randperm(numberOfRows);   
    trainIndexes = randIndexes(1:trainRows);    
    initOfValRows = (trainRows + 1);
    valIndexes = randIndexes(initOfValRows:(initOfValRows + valRows -1));
    initOfTestRows = (initOfValRows + valRows);
    testIndexes = randIndexes(initOfTestRows:(initOfTestRows + testRows-1));

    X_train = X(trainIndexes, :);
    Y_train = Y(:, trainIndexes);
    
    X_val = X(valIndexes, :);
    Y_val = Y(:, valIndexes);
    
    X_test = X(testIndexes, :);
    Y_test = Y(:, testIndexes);
end