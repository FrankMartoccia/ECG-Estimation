clear; clc; close all;

load('data/inputMultiStepRNN', 'net', 'ValidationData', 'PrevisionTarget');

% Initialize a matrix to store RMSE for each test sequence
RmseVector = nan(length(ValidationData), 1);

% Initialize a cell array to store predicted values for each sequence
PredictedValuesCell = cell(length(ValidationData), 1);

% Loop through each test sequence
for windowIdx = 1:length(ValidationData)

    TrueTargets = PrevisionTarget{windowIdx};

    % Reset the network state
    net = resetState(net);

    [net, Y] = predictAndUpdateState(net, ValidationData{windowIdx});

    % Initialize an array to store predicted values
    PredictedValues = nan(size(TrueTargets));

    % Perform closed-loop forecasting
    for t = 1:length(TrueTargets)
        % Predict the next value and update the network state
        [net, Y] = predictAndUpdateState(net, Y);
        PredictedValues(t) = Y;
    end

    % Save predicted values for this sequence
    PredictedValuesCell{windowIdx} = PredictedValues;

    % Calculate the RMSE for this sequence
    error = (PredictedValues - TrueTargets);
    RmseVector(windowIdx) = sqrt(mean(error.^2));

    % Display progress every 500 sequences
    if mod(windowIdx, 500) == 0
        disp(['Processed sequences: ', num2str(windowIdx), ...
            ' - ', num2str(round(windowIdx/length(ValidationData) * 100)), '%']); 
    end
end

% Calculate the overall RMSE
overallRMSE = mean(RmseVector(:));
fprintf('Overall RMSE: %.4f\n', overallRMSE);

save('data/afterMultiStepForecasting', "RmseVector", "overallRMSE", "PredictedValuesCell");

% Plot RMSE for each sequence
figure;
plot(1:length(ValidationData), RmseVector, '.');
xlabel('Sequence Index');
ylabel('RMSE');
title('RMSE for Each Sequence');
grid on;

% Plot histogram of RMSE values
figure;
histogram(RmseVector(:), 50);
xlabel('RMSE');
ylabel('Frequency');
title('Histogram of RMSE Values');
grid on;

% Choose a specific sequence for the single-channel plot
sequenceToPlot = 1; % Change to the desired sequence index

% Extract relevant data for the chosen sequence
trueValues = PrevisionTarget{sequenceToPlot};
predictedValues = PredictedValuesCell{sequenceToPlot};

% Plot the true values and predicted values
figure;
plot(1:length(trueValues), trueValues, 'g', 'LineWidth', 1.5);
hold on;
plot(1:length(predictedValues), predictedValues, 'r', 'LineWidth', 1.5);

xlabel('Time Step');
ylabel('Value');
title(['Comparison of True Values and Predicted Values for Sequence ' num2str(sequenceToPlot)]);
legend('True Values', 'Predicted Values');
grid on;
