function [Features] = extractFeatures(TimeseriesMatrix, method, nWindows)
    if method == "WITHOUT_WIN_METHOD"
        Features = extractFeaturesWithoutWin(TimeseriesMatrix);
    end

    nMeasurements = size(TimeseriesMatrix, 1);

    if method == "CONTIGUOUS_WIN_METHOD"
        windowSize = floor(nMeasurements / nWindows);
        windowStep = windowSize;
        Features = extractFeaturesWithWin(TimeseriesMatrix, nWindows, windowSize, windowStep);
    end

    if method == "OVERLAPPED_WIN_METHOD"
        nContiguousWindows = ceil(nWindows / 2);
        nWindows = nContiguousWindows * 2 - 1;
        windowSize = floor(nMeasurements / nContiguousWindows);
        windowStep = floor(windowSize / 2);
        Features = extractFeaturesWithWin(TimeseriesMatrix, nWindows, windowSize, windowStep);
    end 
end

function [Features] = extractFeaturesWithoutWin(TimeseriesMatrix)
    Features = computeFeatures(TimeseriesMatrix);
end

function [Features] = extractFeaturesWithWin(TimeseriesMatrix, nWindows, windowSize, windowStep)
    N_FEATURE = 13;
    nSignals = size(TimeseriesMatrix,2);
    Features = zeros(1, nWindows * N_FEATURE * nSignals);    
    for windowIndex = 1 :nWindows
        windowStartIndex = (windowIndex - 1) * windowStep + 1;
        windowEndIndex = windowStartIndex + windowSize - 1;
        windowFeatures = computeFeatures(TimeseriesMatrix(windowStartIndex:windowEndIndex, :));
        Features((windowIndex - 1) * N_FEATURE * nSignals + 1 : windowIndex * N_FEATURE * nSignals) = windowFeatures;
    end

end

function [Features] = computeFeatures(TimeseriesMatrix)
    % Computes 13 features from a given signal

    % Feature 1: Mean
    mean_signal = mean(TimeseriesMatrix);

    % Feature 2: Standard deviation
    std_signal = std(TimeseriesMatrix);

    % Feature 3: Maximum amplitude
    max_amp = max(TimeseriesMatrix);

    % Feature 4: Minimum amplitude
    min_amp = min(TimeseriesMatrix);

    % Feature 5: Median
    median_signal = median(TimeseriesMatrix);

    % Feature 6: Interquartile range
    IQR_signal = iqr(TimeseriesMatrix);

    % Feature 7: Skewness
    skewness_signal = skewness(TimeseriesMatrix);

    % Feature 8: Kurtosis
    kurtosis_signal = kurtosis(TimeseriesMatrix);

    % Feature 9: Mean absolute deviation
    mad_signal = mad(TimeseriesMatrix);

    % Feature 10: Root mean square
    rms_signal = rms(TimeseriesMatrix);

    % Feature 11: Peak to peak amplitude
    peak_to_peak_amp = peak2peak(TimeseriesMatrix);

    % Feature 12: Waveform length
    waveform_length = sum(abs(diff(TimeseriesMatrix)));

    % Feature 13: Power spectral density
    [pxx, ~] = pwelch(TimeseriesMatrix);
    psd = bandpower(pxx);

    % Store features in a matrix
    Features = [mean_signal; std_signal; max_amp; min_amp; median_signal;...
                IQR_signal; skewness_signal; kurtosis_signal;...
                mad_signal; rms_signal; peak_to_peak_amp; waveform_length; psd];
    Features = Features(:)';
end

