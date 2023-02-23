function [AugmentedData] = dataAugmentation(nFeaturesMatrixRows, nAugmentations, FeaturesMatrix)
    nFeatures = size(FeaturesMatrix, 2);

    % Create new matrix for augmented data
    AugmentedData = zeros(nFeaturesMatrixRows * (nAugmentations + 1), nFeatures);

    % Loop over each row in original data and generate augmented data
    for i = 1:nFeaturesMatrixRows
        % Add original row to output matrix
        AugmentedData((i-1) * (nAugmentations + 1) + 1, :) = FeaturesMatrix(i, :);

        % Generate nAugmentations samples using random noise addition
        for j = 1:nAugmentations
            % Add random Gaussian noise to each value in row
            noise = normrnd(0, 0.08, [1, nFeatures]);
            augmentedRow = FeaturesMatrix(i, :) + noise;

            if j == 1
                % Check of noise distribution computing RMSE
                rmse = sqrt(mean((FeaturesMatrix(i,:) - augmentedRow).^2));
                disp(['RMSE for row ' num2str(i) ': ' num2str(rmse)]);
            end

            % Add augmented row to output matrix
            AugmentedData((i-1) * (nAugmentations + 1) + j + 1, :) = augmentedRow;

            % Additional visual check for the noise distribution
            % if i == 1 && j == 1
            %     % Plot histogram of added noise
            %     figure;
            %     histogram(noise, 'Normalization', 'probability');
            %     title(sprintf('Histogram of added noise for row %d', i));

            %     % Plot histogram of original data and augmented data
            %     figure;
            %     histogram(FeaturesMatrix(i,:), 'Normalization', 'probability');
            %     hold on;
            %     histogram(augmentedRow, 'Normalization', 'probability');
            %     title(sprintf('Histogram of original and augmented data for row %d', i));
            %     legend('Original Data', 'Augmented Data');
            % end

            % Clear augmentedRow and noise to free memory
            clear augmentedRow noise;
        end
    end
end
