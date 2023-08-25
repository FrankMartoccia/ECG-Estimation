clear
close all
clc

load("data/afterFeaturesSelection.mat");

% Network parameters
hiddenSizes = 60;

% List of training functions to test
trainFunctions = {'trainlm', 'trainrp', 'trainscg'};

% Initialize arrays to store performance results
mseResults = zeros(length(trainFunctions), 1);

% Cross-validation setup
numFolds = 5;
cv = cvpartition(length(InputMeanECG), 'KFold', numFolds);

% Loop through each training function
for i = 1:length(trainFunctions)
    trainFcn = trainFunctions{i};
    
    fprintf('Training with %s...\n', trainFcn);
    
    totalMSE = 0;
    
    % Cross-validation loop
    for fold = 1:numFolds
        trainIdx = cv.training(fold);
        testIdx = cv.test(fold);
        
        trainData = InputMeanECG(:, trainIdx);
        trainTarget = TargetMeanECG(:, trainIdx);
        testData = InputMeanECG(:, testIdx);
        testTarget = TargetMeanECG(:, testIdx);
        
        % Network creation
        net = fitnet(hiddenSizes, trainFcn);
        net.divideParam.trainRatio = 0.7;
        net.divideParam.valRatio = 0.1;
        net.divideParam.testRatio = 0.2;
        net.performFcn = 'mse'; 
        
        [net, tr] = train(net, trainData, trainTarget);
        
        % Evaluate performance on test data
        y = net(testData);
        mse = perform(net, testTarget', y);
        totalMSE = totalMSE + mse;
    end
    
    % Calculate average MSE for this training function
    avgMSE = totalMSE / numFolds;
    mseResults(i) = avgMSE;
    
    fprintf('%s - Average MSE: %.4f\n', trainFcn, avgMSE);
end

% Compare performance of different training functions
figure;
bar(mseResults);
xticks(1:length(trainFunctions));
xticklabels(trainFunctions);
ylabel('Average MSE');
title('Comparison of Training Algorithms');