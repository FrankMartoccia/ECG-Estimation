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

FeaturesWithoutWin = zeros(N_FEATURES_MATRIX_ROWS, N_FEATURES_MATRIX_COLUMNS);

FeaturesContiguousWin = zeros(N_FEATURES_MATRIX_ROWS,...
    N_FEATURES_MATRIX_COLUMNS * N_CONTIGUOUS_WIN_NUM);

FeaturesOverlappedWin = zeros(N_FEATURES_MATRIX_ROWS,...
    N_FEATURES_MATRIX_COLUMNS * N_OVERLAPPED_WIN_NUM);


FileList = dir(fullfile(FOLDER_PATH, '*.csv'));

timeseriesCounter = 0;

for m = 1:length(FileList)
    fileName = FileList(m).name;
    
    if (contains(fileName, 'timeseries'))
        timeseriesCounter = timeseriesCounter + 1;
        disp(timeseriesCounter);
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


    end
    
end