clear; clc; close all; 

load('data/inputCNN.mat');

% Definition of the CNN structure

nFilters = 8;
FiltersSize = 16;

layers = [
    sequenceInputLayer(11)
    
    convolution1dLayer(FiltersSize, numFilters, 'Stride', 2, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling1dLayer(2, 'Stride', 2, 'Padding', 'same')


    convolution1dLayer(FiltersSize / 2, 2 * numFilters, 'Stride', 2, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling1dLayer(2, 'Stride', 2, 'Padding', 'same')

    globalAveragePooling1dLayer

    fullyConnectedLayer(50)
    fullyConnectedLayer(1)

    regressionLayer
];


