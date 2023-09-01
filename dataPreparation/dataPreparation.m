clear

% CONSTANTS
FOLDER_PATH = 'dataset';

WITHOUT_WIN_METHOD = "WITHOUT_WIN_METHOD";
CONTIGUOUS_WIN_METHOD = "CONTIGUOUS_WIN_METHOD";
OVERLAPPED_WIN_METHOD = "OVERLAPPED_WIN_METHOD";

N_FEATURES = 13;
N_FEATURES_MATRIX_ROWS = 66; % n. of subjects * n. of activities
N_FEATURES_MATRIX_COLUMNS = 11 * N_FEATURES; % n. of signals * n. of features
N_CONTIGUOUS_WIN_NUM = 4; % n. of contiguous windows
N_OVERLAPPED_WIN_NUM = 7; % n. of overlapped windows
N_AUGMENTATIONS = 50;
N_SELECTED_FEATURES = 10;

FeaturesWithoutWin = zeros(N_FEATURES_MATRIX_ROWS, N_FEATURES_MATRIX_COLUMNS);

FeaturesContiguousWin = zeros(N_FEATURES_MATRIX_ROWS,...
    N_FEATURES_MATRIX_COLUMNS * N_CONTIGUOUS_WIN_NUM);

FeaturesOverlappedWin = zeros(N_FEATURES_MATRIX_ROWS,...
    N_FEATURES_MATRIX_COLUMNS * N_OVERLAPPED_WIN_NUM);

TargetMeanECG = zeros(N_FEATURES_MATRIX_ROWS, 1);

TargetStdECG = zeros(N_FEATURES_MATRIX_ROWS, 1);

TargetActivity = zeros(N_FEATURES_MATRIX_ROWS, 3);

FileList = dir(fullfile(FOLDER_PATH, '*.csv'));

timeseriesCounter = 0;
targetCounter = 0;

for m = 1:length(FileList)
    fileName = FileList(m).name;
    
    if (contains(fileName, 'timeseries'))
        timeseriesCounter = timeseriesCounter + 1;
        disp(['Timeseries nr. ' num2str(timeseriesCounter)]);
        TimeseriesTable = readtable(fullfile(FOLDER_PATH, fileName), 'Range', 'B:L');
        TimeseriesMatrix = table2array(TimeseriesTable);
        
        % Check outliers
        if (max(isoutlier(TimeseriesMatrix)) == 1) 
            disp('Outlier detected')
        end
        % Detect and replace outliers 
        TimeseriesMatrix = filloutliers(TimeseriesMatrix, 'linear');

        % Feature extraction (without window)
        FeaturesWithoutWin(timeseriesCounter, :) = extractFeatures(TimeseriesMatrix, WITHOUT_WIN_METHOD); 
        
        % Feature extraction (with contiguous windows)
        FeaturesContiguousWin(timeseriesCounter, :) = extractFeatures(TimeseriesMatrix,...
            CONTIGUOUS_WIN_METHOD, N_CONTIGUOUS_WIN_NUM); 
        
        % Feature extraction (with overlapped windows)
        FeaturesOverlappedWin(timeseriesCounter, :) = extractFeatures(TimeseriesMatrix,...
            OVERLAPPED_WIN_METHOD, N_OVERLAPPED_WIN_NUM);         

    else
        targetCounter = targetCounter + 1;
        TargetTable = readtable(fullfile(FOLDER_PATH, fileName), 'Range', 'B:B');
        TargetECG = table2array(TargetTable);
        TargetMeanECG(targetCounter, 1) = mean(TargetECG);
        TargetStdECG(targetCounter, 1) = std(TargetECG);
        
        if contains(fileName, "run")
            TargetActivity(targetCounter, :) = [1, 0, 0];
        elseif contains(fileName, "sit")
            TargetActivity(targetCounter, :) = [0, 1, 0];
        else
            TargetActivity(targetCounter, :) = [0, 0, 1];
        end
    end
    
end

save('data/beforeNormalization', 'FeaturesWithoutWin', 'FeaturesContiguousWin', ...
    'FeaturesOverlappedWin', 'TargetMeanECG', 'TargetStdECG', 'TargetActivity');

% Features normalization and elimination of the correlated features
FeaturesWithoutWin = normalizeFeatures(FeaturesWithoutWin);
FeaturesWithoutWin = deleteCorrelatedFeatures(FeaturesWithoutWin);

FeaturesContiguousWin = normalizeFeatures(FeaturesContiguousWin);
FeaturesContiguousWin = deleteCorrelatedFeatures(FeaturesContiguousWin);

FeaturesOverlappedWin = normalizeFeatures(FeaturesOverlappedWin);
FeaturesOverlappedWin = deleteCorrelatedFeatures(FeaturesOverlappedWin);

save('data/afterNormalization', 'FeaturesWithoutWin', 'FeaturesContiguousWin', ...
    'FeaturesOverlappedWin', 'TargetMeanECG', 'TargetStdECG', 'TargetActivity');

% Data augmentation on the extracted features
FeaturesWithoutWin = normalizeFeatures(dataAugmentation(N_FEATURES_MATRIX_ROWS,...
 N_AUGMENTATIONS, FeaturesWithoutWin));
FeaturesContiguousWin = normalizeFeatures(dataAugmentation(N_FEATURES_MATRIX_ROWS,...
 N_AUGMENTATIONS, FeaturesContiguousWin));
FeaturesOverlappedWin = normalizeFeatures(dataAugmentation(N_FEATURES_MATRIX_ROWS,...
 N_AUGMENTATIONS, FeaturesOverlappedWin));

% Adapting Target matrices to data augmentation
TargetMeanECG = repelem(TargetMeanECG, N_AUGMENTATIONS + 1, 1);
TargetStdECG = repelem(TargetStdECG, N_AUGMENTATIONS + 1, 1);
TargetActivity = repelem(TargetActivity, N_AUGMENTATIONS + 1, 1);

save('data/afterDataAugmentation', 'FeaturesWithoutWin', 'FeaturesContiguousWin', ...
    'FeaturesOverlappedWin', 'TargetMeanECG', 'TargetStdECG', 'TargetActivity');

% Feature selection options
options = statset('Display', 'iter','UseParallel',true);

% Running feature selection and saving results on file (no windows)
% Mean
selectFeatures(FeaturesWithoutWin, TargetMeanECG, options, 'MeanWithoutWin', N_SELECTED_FEATURES);
% Std
selectFeatures(FeaturesWithoutWin, TargetStdECG, options, 'StdWithoutWin', N_SELECTED_FEATURES);

% Running feature selection and saving results on file (contiguous windows)
% Mean
selectFeatures(FeaturesContiguousWin, TargetMeanECG, options, 'MeanContiguousWin', N_SELECTED_FEATURES);
% Std
selectFeatures(FeaturesContiguousWin, TargetStdECG, options, 'StdContiguousWin', N_SELECTED_FEATURES);

% Running feature selection and saving results on file (overlapped windows)
% Mean
selectFeatures(FeaturesOverlappedWin, TargetMeanECG, options, 'MeanOverlappedWin', N_SELECTED_FEATURES);
% Std
selectFeatures(FeaturesOverlappedWin, TargetStdECG, options, 'StdOverlappedWin', N_SELECTED_FEATURES);

% The best results were obtained by extracting features without windows
% Extraction of the best features columns from file
BestFeaturesMeanECG = importdata('results/dataPreparation/sfMeanWithoutWinColumns.txt', ' ');
BestFeaturesStdECG = importdata('results/dataPreparation/sfStdWithoutWinColumns.txt', ' ');
BestFeaturesUnion = unique(sort([BestFeaturesMeanECG; BestFeaturesStdECG]));

% Creation of input matrices with only the best features
InputMeanECG = FeaturesWithoutWin(:, BestFeaturesMeanECG)';
InputStdECG = FeaturesWithoutWin(:, BestFeaturesStdECG)';
% Creation of target matrices
TargetMeanECG = TargetMeanECG';
TargetStdECG = TargetStdECG';

% Creation of input matrices and vectors for activity recognition
InputActivity = FeaturesWithoutWin(:, BestFeaturesUnion)';
TargetActivityClasses = vec2ind(TargetActivity');
TargetActivityClassesVec = TargetActivity';

save('data/afterFeaturesSelection', 'InputMeanECG', 'InputStdECG', 'TargetMeanECG', ...
    'TargetStdECG', 'InputActivity', 'TargetActivityClasses', 'TargetActivityClassesVec');

% Normalization of the features values in [0, 1] (Min-max normalization)
function [NormalizedFeatures] = normalizeFeatures(InputFeatures)
    minValues = min(InputFeatures);
    NormalizedFeatures = (InputFeatures - minValues) ./ (max(InputFeatures) - minValues);
end

% Removal of all features with correlation above the specified threshold
function [Features] = deleteCorrelatedFeatures(InputFeatures)
    CorrelationMatrix = corr(InputFeatures);
    % find highly correlated features
    CorrelatedFeatures = abs(triu(CorrelationMatrix, 1)) > 0.9;
    % remove highly correlated features
    Features = InputFeatures(:, all(~CorrelatedFeatures));
end

% Run feature selection and save the results
function selectFeatures(Features, Target, options, fileName, nFeatures)
    filePath = (strcat('results/dataPreparation/sf', fileName));
    diary(strcat(filePath, 'Log.txt'));
    selectedFeatures = sequentialfs(@selectionCriterion, Features, Target, ...
        'options', options, 'nfeatures', nFeatures);
    diary('off');
    saveBestFeatures(filePath, selectedFeatures);
end

% Save the values of the selected features columns
function saveBestFeatures(filePath, selectedFeatures)
    filePath = fopen(strcat(filePath, 'Columns.txt'), 'w');
    fprintf(filePath,'%s\n', num2str(find(selectedFeatures)));
    fclose(filePath);
end