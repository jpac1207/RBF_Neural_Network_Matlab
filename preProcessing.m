% Esta função utiliza os mapas contidos no parâmetro 'preProcessingConfig'
% de forma a converter cada elemento contido em 'data' para um escalar correpondente. 
% Os valores de retorno representam os padrões de entrada e de saída
% convertidos para os escalares respectivos
function [processedData, processedLabels] = preProcessing(data, preProcessingConfig)  
    %rows = size(data, 1);
    rows = 10;
    dataMatrix = zeros(rows, size(data, 2));
    labels = zeros(4, rows);
    for i = 1:rows        
        cellArray = strsplit(data{i}, ',');        
        buying = preProcessingConfig.buyingMap(cellArray{1});
        maint = preProcessingConfig.maintMap(cellArray{2});
        doors = preProcessingConfig.doorsMap(cellArray{3});
        persons = preProcessingConfig.personsMap(cellArray{4});
        lugBoot = preProcessingConfig.lugBootMap(cellArray{5});
        safety = preProcessingConfig.safetyMap(cellArray{6});
        label = preProcessingConfig.labelMap(cellArray{7});
        dataMatrix(i, 1) = buying;
        dataMatrix(i, 2) = maint;
        dataMatrix(i, 3) = doors;
        dataMatrix(i, 4) = persons;
        dataMatrix(i, 5) = lugBoot;
        dataMatrix(i, 6) = safety;
        for j = 1:4
            if(j == label)
                labels(j, i) = 1;
            else
                labels(j, i) = 0;
            end
        end       
    end
    processedData = dataMatrix;
    processedLabels = labels;
end