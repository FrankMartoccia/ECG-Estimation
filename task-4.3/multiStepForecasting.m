clear; clc; close all;

load('data/inputMultiStepRNN', 'net', 'ValidationData', 'PrevisionTarget');

% Initialize a matrix to store RMSE for each test sequence
RmseMatrix = nan(length(ValidationData), 1);

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
    RmseMatrix(windowIdx) = sqrt(mean(error.^2));

    % Display progress every 500 sequences
    if mod(windowIdx, 500) == 0
        disp(['Processed sequence ', num2str(windowIdx), ...
            ' - ', num2str(round(windowIdx/length(ValidationData) * 100)), '%']); 
    end
end

% Calculate the overall RMSE
overallRMSE = mean(RmseMatrix(:));
fprintf('Overall RMSE: %.4f\n', overallRMSE);

% Plot RMSE for each sequence
figure;
plot(1:length(ValidationData), RmseMatrix, '.');
xlabel('Sequence Index');
ylabel('RMSE');
title('RMSE for Each Sequence');
grid on;

% Plot histogram of RMSE values
figure;
histogram(RmseMatrix(:), 50);
xlabel('RMSE');
ylabel('Frequency');
title('Histogram of RMSE Values');
grid on;

% Choose a specific sequence for the single-channel plot
sequenceToPlot = 1; % Change to the desired sequence index

% Plot the input data, true values, and predicted values
figure;
plot(1:length(ValidationData{sequenceToPlot}), ValidationData{sequenceToPlot}, ...
    'b', 'LineWidth', 1.5);

hold on;

plot(length(ValidationData{sequenceToPlot}):length(ValidationData{sequenceToPlot}) ...
    + length(PrevisionTarget{sequenceToPlot}) - 1, PrevisionTarget{sequenceToPlot}, ...
    'g', 'LineWidth', 1.5);

plot(length(ValidationData{sequenceToPlot}):length(ValidationData{sequenceToPlot}) ...
    + length(PrevisionTarget{sequenceToPlot}) - 1, PredictedValuesCell{sequenceToPlot}, ...
    'r.', 'MarkerSize', 10);

xlabel('Time Step');
ylabel('Value');
title(['Example of Prediction for Sequence ' num2str(sequenceToPlot)]);
legend('Input Data', 'True Values', 'Predicted Values');
grid on;
