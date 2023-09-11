clear; clc; close all;

% Constants
WINDOW_SIZE = 20;

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
cv = cvpartition(length(TargetRNN), 'Holdout', 0.2);

trainIdx = training(cv);
validationIdx = test(cv);

TrainData = InputWindow(trainIdx);
TrainTarget = TargetRNN(trainIdx);
ValidationData = InputWindow(validationIdx);
ValidationTarget = TargetRNN(validationIdx);

% Definition of training options
options = trainingOptions('adam', ...
    MaxEpochs = 15, ...
    MiniBatchSize = 1500, ...
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

% Definition of the RNN structure
layers = [
    sequenceInputLayer(1)
    lstmLayer(80, 'OutputMode', 'last')
    fullyConnectedLayer(1)
    regressionLayer
];

net = trainNetwork(TrainData, TrainTarget, layers, options); 

net = resetState(net);

yTest = predict(net, ValidationData, ExecutionEnvironment ='auto', MiniBatchSize = 100);

figure;
plot(yTest(1:100),'--');
hold on;
plot(ValidationTarget(1:100));
hold off;

%% Compute and analyze error
% To evaluate the accuracy, calculate the root mean squared error (RMSE) 
% between the predictions and the target.
rmse = sqrt(mse(yTest, ValidationTarget));

% Calculate the mean RMSE over all test observations.
fprintf("RMSE on validation set: %f\n", rmse);
