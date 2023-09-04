clear; clc; close all; 

load('data/inputCNN.mat');

nFilters = 64;
filtersSize = 64;
fclSize = 100;

% Definition of the CNN structure
layers = [
    sequenceInputLayer(11)
    
    convolution1dLayer(15, nFilters, 'Stride', 2, 'Padding', 'same')
    batchNormalizationLayer
    leakyReluLayer
    maxPooling1dLayer(4, 'Stride', 4, 'Padding', 'same')

    convolution1dLayer(11 , 2 * nFilters, 'Stride', 2, 'Padding', 'same')
    batchNormalizationLayer
    leakyReluLayer
    maxPooling1dLayer(4, 'Stride', 4, 'Padding', 'same')

    convolution1dLayer(9 , 3 * nFilters, 'Stride', 2, 'Padding', 'same')
    batchNormalizationLayer
    leakyReluLayer
    maxPooling1dLayer(4, 'Stride', 4, 'Padding', 'same')

    convolution1dLayer(9 , 4 * nFilters, 'Stride', 2, 'Padding', 'same')
    batchNormalizationLayer
    leakyReluLayer
    maxPooling1dLayer(4, 'Stride', 4, 'Padding', 'same')

    globalAveragePooling1dLayer

    fullyConnectedLayer(fclSize)
    leakyReluLayer

    fullyConnectedLayer(1)

    regressionLayer
];

% Creation of training and validation sets
cv = cvpartition(length(TargetCNN), 'Holdout', 0.2);

trainIdx = training(cv);
validationIdx = test(cv);

TrainData = TimeseriesCNN(trainIdx);
TrainTarget = TargetCNN(trainIdx);
ValidationData = TimeseriesCNN(validationIdx);
ValidationTarget = TargetCNN(validationIdx);

% Definition of training options
options = trainingOptions('adam', ...
    MaxEpochs = 30, ...
    MiniBatchSize = 80, ...
    Shuffle = 'every-epoch' , ...
    InitialLearnRate = 0.01, ...
    LearnRateSchedule = 'piecewise', ...
    LearnRateDropPeriod = 10, ...
    LearnRateDropFactor = 0.1, ...
    L2Regularization = 0.01, ...
    ValidationData =  {ValidationData ValidationTarget}, ...
    ValidationFrequency = 30, ...
    ExecutionEnvironment = 'auto', ...
    Plots = 'training-progress', ...
    Verbose = 1, ...
    VerboseFrequency = 1 ...
);

net = trainNetwork(TrainData, TrainTarget, layers, options); 

% Test against validation set
yTest = predict(net, ValidationData, ExecutionEnvironment ='auto', MiniBatchSize = 100);
figure; 
plotregression(ValidationTarget, yTest);

% Test against training set
yTrain = predict(net, TrainData, ExecutionEnvironment ='auto', MiniBatchSize = 100);
figure;
plotregression(TrainTarget, yTrain); 