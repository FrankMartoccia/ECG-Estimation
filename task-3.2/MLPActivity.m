% Workspace cleaning
clear
close all
clc

load("data/afterFeaturesSelection.mat");

% Network parameters
hiddenSizes = 10;
trainFcn = 'trainbr'; % Tested trainbr, trainlm and trainrp

% Network creation
net = patternnet(hiddenSizes, trainFcn);
net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio = 0.0;
net.divideParam.testRatio = 0.2;
net.performFcn = 'mse';
% performFcn = 'crossentropy' (by default)

net = train(net, InputActivity, TargetActivityClassesVec);
y = net(InputActivity);

% Plot creation
figure, plotconfusion(TargetActivityClassesVec, y)
