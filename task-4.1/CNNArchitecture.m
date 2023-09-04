clear; clc; close all; 

load('data/inputCNN.mat');

% Definition of values for hyperparameters
nFilters = [8, 16, 32];
filtersSize = [16, 32, 64];
fclSize = [50, 100];

hyperparameterCombinations = combvec(nFilters, filtersSize, fclSize)';

mseResults = zeros(length(hyperparameterCombinations), 1);

numFolds = 5; 
cv = cvpartition(length(TargetCNN), 'KFold', numFolds);

% Loop on each combination of hyperparameters
for combination = 1:length(hyperparameterCombinations)
    hyperparams = hyperparameterCombinations(combination, :);

    currentNFilters = hyperparams(1);
    currentFiltersSize = hyperparams(2);
    currentFclSize = hyperparams(3);

    % Definition of the CNN structure
    layers = [
        sequenceInputLayer(11)
        
        convolution1dLayer(currentFiltersSize, currentNFilters, 'Stride', 2, 'Padding', 'same')
        batchNormalizationLayer
        reluLayer
        maxPooling1dLayer(2, 'Stride', 2, 'Padding', 'same')
    
        convolution1dLayer(currentFiltersSize / 2, 2 * currentNFilters, 'Stride', 2, 'Padding', 'same')
        batchNormalizationLayer
        reluLayer
        maxPooling1dLayer(2, 'Stride', 2, 'Padding', 'same')
    
        globalAveragePooling1dLayer
    
        fullyConnectedLayer(currentFclSize)
        fullyConnectedLayer(1)
    
        regressionLayer
    ];

    totalMSE = 0;
                
    % Cross-validation loop
    for fold = 1:numFolds
        trainIdx = cv.training(fold);
        validationIdx = cv.test(fold);
        
        TrainData = TimeseriesCNN(trainIdx);
        TrainTarget = TargetCNN(trainIdx);
        ValidationData = TimeseriesCNN(validationIdx);
        ValidationTarget = TargetCNN(validationIdx);
        
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

        y = predict(net, ValidationData);
        y = double(y);
        mse = immse(y, ValidationTarget);
        totalMSE = totalMSE + mse;
    end

    % Compute average MSE for this hyperparameter combination
    avgMSE = totalMSE / numFolds;
    mseResults(combination) = avgMSE;
    
    fprintf('nFilters=%d, filtersSize=%d, fclSize=%d - Average MSE: %.4f\n', ...
        nFilters, filtersSize, fclSize, avgMSE);
    
end

% Compare performance of different training functions
combinationLabels = cellstr(num2str((1:length(hyperparameterCombinations))'));
figure;
bar(mseResults);
xticks(1:length(hyperparameterCombinations));
xticklabels(combinationLabels);
xlabel('Combination ID');
ylabel('Average MSE');
title('Comparison of combinations of hyperparameters');