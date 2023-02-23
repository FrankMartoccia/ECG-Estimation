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
        % Remove outliers
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

FeaturesWithoutWin = normalizeFeatures(FeaturesWithoutWin);
FeaturesWithoutWin = deleteCorrelatedFeatures(FeaturesWithoutWin);

FeaturesContiguousWin = normalizeFeatures(FeaturesContiguousWin);
FeaturesContiguousWin = deleteCorrelatedFeatures(FeaturesContiguousWin);

FeaturesOverlappedWin = normalizeFeatures(FeaturesOverlappedWin);
FeaturesOverlappedWin = deleteCorrelatedFeatures(FeaturesOverlappedWin);

save('data/afterNormalization', 'FeaturesWithoutWin', 'FeaturesContiguousWin', ...
    'FeaturesOverlappedWin', 'TargetMeanECG', 'TargetStdECG', 'TargetActivity');

FeaturesWithoutWin = normalizeFeatures(dataAugmentation(N_FEATURES_MATRIX_ROWS,...
 N_AUGMENTATIONS, FeaturesWithoutWin));
FeaturesContiguousWin = normalizeFeatures(dataAugmentation(N_FEATURES_MATRIX_ROWS,...
 N_AUGMENTATIONS, FeaturesContiguousWin));
FeaturesOverlappedWin = normalizeFeatures(dataAugmentation(N_FEATURES_MATRIX_ROWS,...
 N_AUGMENTATIONS, FeaturesOverlappedWin));

TargetMeanECG = repelem(TargetMeanECG, N_AUGMENTATIONS, 1);
TargetStdECG = repelem(TargetStdECG, N_AUGMENTATIONS, 1);
TargetActivity = repelem(TargetActivity, N_AUGMENTATIONS, 1);

save('data/afterDataAugmentation', 'FeaturesWithoutWin', 'FeaturesContiguousWin', ...
    'FeaturesOverlappedWin', 'TargetMeanECG', 'TargetStdECG', 'TargetActivity');


function [NormalizedFeatures] = normalizeFeatures(InputFeatures)
    minValues = min(InputFeatures);
    
    NormalizedFeatures = (InputFeatures - minValues) ./ (max(InputFeatures) - minValues);
end

function [Features] = deleteCorrelatedFeatures(InputFeatures)
    CorrelationMatrix = corr(InputFeatures);
    % find highly correlated features
    CorrelatedFeatures = abs(triu(CorrelationMatrix, 1)) > 0.9;
    % remove highly correlated features
    Features = InputFeatures(:, all(~CorrelatedFeatures));
end
