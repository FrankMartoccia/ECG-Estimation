clear; clc; close all;

% Constants
WINDOW_SIZE = 40;

load('data/inputRNN.mat');

% The network takes in input the measurements of one file at a time
% (The ones corresponding to the current index)
InputRNN = InputRNN{1};

% Data normalization
InputRNN = normalize(InputRNN,'range');

InputWindow = cell(length(InputRNN) - WINDOW_SIZE, 1);
TargetRNN = zeros(length(InputRNN) - WINDOW_SIZE, 1);

for startIdx = 1:(length(InputRNN) - WINDOW_SIZE)
    endIdx = startIdx + WINDOW_SIZE - 1;
    
    InputWindow{startIdx} = InputRNN(startIdx:endIdx);
    TargetRNN(startIdx) = InputRNN(endIdx + 1);
end

% Creation of training and validation sets
nMeasurements = size(InputWindow, 1);
trainIdx = 1: floor(0.8 * nMeasurements);
validationIdx = floor(0.8 * nMeasurements) + 1 :nMeasurements;

TrainData = InputWindow(trainIdx);
TrainTarget = TargetRNN(trainIdx);
ValidationData = InputWindow(validationIdx);
ValidationTarget = TargetRNN(validationIdx);

% Definition of training options
options = trainingOptions('adam', ...
    MaxEpochs = 10, ...
    MiniBatchSize = 1500, ...
    Shuffle = 'every-epoch' , ...
    InitialLearnRate = 0.01, ...
    LearnRateSchedule = 'piecewise', ...
    LearnRateDropPeriod = 10, ...
    LearnRateDropFactor = 0.1, ...
    ValidationData =  {ValidationData ValidationTarget}, ...
    ValidationFrequency = 30, ...
    ExecutionEnvironment = 'auto', ...
    Plots = 'training-progress', ...
    SequencePaddingDirection = 'left', ...
    Verbose = 1, ...
    VerboseFrequency = 1 ...
);

% Definition of the RNN structure
layers = [
    sequenceInputLayer(1)
    lstmLayer(80, 'OutputMode', 'last')
    fullyConnectedLayer(1)
    regressionLayer
];

net = trainNetwork(TrainData, TrainTarget, layers, options); 

net = resetState(net);

yTest = predict(net, ValidationData, ExecutionEnvironment ='auto');

% Calculate RMSE
rmse = sqrt(mean((yTest - ValidationTarget).^2));
fprintf('RMSE: %.4f\n', rmse);

% Plot of predicted vs correct values of ECG
figure;
plot(yTest(1:150),'--');
hold on;
plot(ValidationTarget(1:150));
hold off;
