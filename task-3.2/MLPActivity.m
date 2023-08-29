% Workspace cleaning
clear
close all
clc

load("data/afterFeaturesSelection.mat");

% Network parameters
hiddenSizes = 10; % Tested with 7,8,9,10 neurons
trainFcn = 'trainbr'; % Tested trainbr, trainlm and trainrp

% Network creation
net = patternnet(hiddenSizes, trainFcn);
net.divideParam.trainRatio = 0.8;
% trainbr doesn't need a validation set
net.divideParam.valRatio = 0.0; % It was set to 0.1 for trainlm and trainrp
net.divideParam.testRatio = 0.2; % It was set to 0.1 for trainlm and trainrp
net.performFcn = 'mse';
% performFcn = 'crossentropy' (by default)

net = train(net, InputActivity, TargetActivityClassesVec);
y = net(InputActivity);

% Plot creation
figure, plotconfusion(TargetActivityClassesVec, y)
