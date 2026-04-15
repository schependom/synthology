% --- Experiment 2/3: Hop Distribution Bar Charts (Auto from run artifacts) ---
% This script reads dataset paths from the latest compare summaries:
%   reports/experiment_runs/*/exp2/report_data/*/report/summary.json
%   reports/experiment_runs/*/exp3/report_data/*/report/summary.json
% and computes positive inferred hop buckets from train/targets.csv.

C = kulcolors();

set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultAxesFontSize', 16);
set(groot, 'defaultLegendFontSize', 14);
set(groot, 'defaultTextFontSize', 14);

repoRoot = fileparts(fileparts(mfilename('fullpath')));
outDir = fullfile(repoRoot, 'paper', 'figures');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

% Optional explicit run pinning (leave empty to auto-pick latest summaries).
exp2SummaryOverride = '';
exp3SummaryOverride = '';

if strlength(string(exp2SummaryOverride)) > 0
    exp2Summary = exp2SummaryOverride;
else
    exp2Summary = find_latest_summary(fullfile(repoRoot, 'reports', 'experiment_runs', '*', 'exp2', 'report_data', '*', 'report', 'summary.json'));
end
if isempty(exp2Summary)
    error('No Exp2 compare summary found. Run "uv run invoke exp2-report-data" first.');
end

[exp2Data, exp2Labels] = load_bucket_data(exp2Summary, repoRoot);
render_hop_chart(exp2Data, exp2Labels, C, ...
    'Experiment 2: Hop Distribution of Positive Inferred Targets', ...
    fullfile(outDir, 'exp2_hop_distr.pdf'));

if strlength(string(exp3SummaryOverride)) > 0
    exp3Summary = exp3SummaryOverride;
else
    exp3Summary = find_latest_summary(fullfile(repoRoot, 'reports', 'experiment_runs', '*', 'exp3', 'report_data', '*', 'report', 'summary.json'));
end
if ~isempty(exp3Summary)
    [exp3Data, exp3Labels] = load_bucket_data(exp3Summary, repoRoot);
    render_hop_chart(exp3Data, exp3Labels, C, ...
        'Experiment 3: Hop Distribution of Positive Inferred Targets', ...
        fullfile(outDir, 'exp3_hop_distr.pdf'));
else
    fprintf('[WARN] No Exp3 compare summary found. Skipping exp3_hop_distr.pdf\n');
end


function summaryPath = find_latest_summary(pattern)
    matches = dir(pattern);
    if isempty(matches)
        summaryPath = '';
        return;
    end
    [~, idx] = max([matches.datenum]);
    summaryPath = fullfile(matches(idx).folder, matches(idx).name);
end


function [data, labels] = load_bucket_data(summaryPath, repoRoot)
    summary = jsondecode(fileread(summaryPath));
    methodEntries = summary.methods;

    baselinePath = '';
    synthologyPath = '';

    for i = 1:numel(methodEntries)
        methodName = lower(string(methodEntries(i).method));
        methodPath = string(methodEntries(i).path);

        if startsWith(methodName, 'base')
            baselinePath = methodPath;
        elseif startsWith(methodName, 'synth')
            synthologyPath = methodPath;
        end
    end

    if strlength(baselinePath) == 0 || strlength(synthologyPath) == 0
        error('Could not resolve baseline/synthology method paths from %s', summaryPath);
    end

    baselineTargets = resolve_targets_path(char(baselinePath), repoRoot);
    synthologyTargets = resolve_targets_path(char(synthologyPath), repoRoot);

    baselineBuckets = hop_buckets_from_targets(baselineTargets);
    synthologyBuckets = hop_buckets_from_targets(synthologyTargets);

    data = [baselineBuckets(:), synthologyBuckets(:)];
    labels = {'UDM Baseline', 'Synthology'};
end


function targetsPath = resolve_targets_path(datasetPath, repoRoot)
    if is_absolute_path(datasetPath)
        basePath = datasetPath;
    else
        basePath = fullfile(repoRoot, datasetPath);
    end
    targetsPath = fullfile(basePath, 'train', 'targets.csv');
    if ~isfile(targetsPath)
        error('Missing targets.csv at %s', targetsPath);
    end
end


function out = hop_buckets_from_targets(targetsCsv)
    T = readtable(targetsCsv, 'TextType', 'string');

    labelsNum = numeric_col(T.label);
    hopsNum = numeric_col(T.hops);

    if ismember('type', T.Properties.VariableNames)
        types = lower(string(T.type));
    else
        types = repmat("", height(T), 1);
    end

    inferredMask = startsWith(types, 'inf') | (types == "inferred");
    positiveMask = labelsNum == 1;
    keep = inferredMask & positiveMask & ~isnan(hopsNum) & (hopsNum >= 1);
    hops = hopsNum(keep);

    out = [
        sum(hops == 1), ...
        sum(hops == 2), ...
        sum(hops == 3), ...
        sum(hops >= 4)
    ];
end


function render_hop_chart(data, labels, C, titleText, outFile)
    categories = {'1 Hop', '2 Hops', '3 Hops', '\geq 4 Hops'};
    fig = figure('Position', [100, 100, 900, 340]);

    b = bar(data, 'grouped');
    b(1).FaceColor = C.KULijsblauw;
    b(2).FaceColor = C.KULcorporate;

    set(gca, 'XTickLabel', categories, 'FontSize', 12);
    ylabel('Number of Positive Inferred Facts', 'FontSize', 13, 'FontWeight', 'bold');
    xlabel('Logical Proof Depth (Hops)', 'FontSize', 13, 'FontWeight', 'bold');
    title(titleText, 'FontSize', 14, 'FontWeight', 'bold');

    legend(labels, 'Location', 'northeast', 'FontSize', 12, 'Interpreter', 'latex');
    box off;
    grid on;
    set(gca, 'GridLineStyle', ':', 'GridAlpha', 0.6);

    exportgraphics(fig, outFile, 'ContentType', 'vector');
    fprintf('Saved hop chart: %s\n', outFile);
    close(fig);
end


function tf = is_absolute_path(pathStr)
    tf = startsWith(pathStr, '/') || startsWith(pathStr, '~') || ~isempty(regexp(pathStr, '^[A-Za-z]:[\\/]', 'once'));
end


function nums = numeric_col(col)
    if isnumeric(col)
        nums = double(col);
        return;
    end
    nums = str2double(string(col));
end
