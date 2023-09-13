clear; clc; close all;

% CONSTANTS
FOLDER_PATH = 'dataset';
N_MEASUREMENTS = 66;

InputRNN = cell(N_MEASUREMENTS, 1);

FileList = dir(fullfile(FOLDER_PATH, '*targets.csv'));

for m = 1:length(FileList)
    fileName = FileList(m).name;
    disp(fileName)
    InputTable = readtable(fullfile(FOLDER_PATH, fileName), 'Range', 'B:B');
    InputECG = table2array(InputTable); 
    InputRNN{m} = InputECG';
end

save('data/inputRNN.mat', 'InputRNN');