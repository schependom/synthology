% exp23_hop_distribution.m
%
% Reads the pre-computed hops_by_method.csv produced by the data reporter
% and renders semilogy grouped bar charts for Exp2 and Exp3.
%
% Hops are binned: 1-3 (easy), 4, 5, 6, 7, 8+ (hard).

disp('IMPORTANT: delete the existing PDF files before running this script!!\n');

% --- CONFIGURATION ---
FS = 24;           % Font size scale
relative = false;  % Set to true for density (sum to 1), false for absolute counts
% ---------------------

EXP2_REPORT_DIR = '/dtu/blackhole/16/221590/synthology/reports/experiment_runs/2026-04-18/exp2/report_data/185853_compare/report';
EXP3_REPORT_DIR = '/dtu/blackhole/16/221590/synthology/reports/experiment_runs/2026-04-22/exp3/report_data/185842_compare/report';

% ---------------------------------------------------------------------------

% Apply common styling and retrieve the color palette
style = common(FS);
C = style.C;

repoRoot = fileparts(fileparts(mfilename('fullpath')));
outDir   = fullfile(repoRoot, 'paper', 'figures');
if ~exist(outDir, 'dir'), mkdir(outDir); end

% Render charts passing the font size and relative flag
render_hop_chart( ...
    fullfile(EXP2_REPORT_DIR, 'hops_by_method.csv'), ...
    'Experiment 2: Hop Distribution (Family Tree)', ...
    fullfile(outDir, 'exp2_hop_distr.pdf'), C, FS, relative);

render_hop_chart( ...
    fullfile(EXP3_REPORT_DIR, 'hops_by_method.csv'), ...
    'Experiment 3: Hop Distribution (OWL2Bench)', ...
    fullfile(outDir, 'exp3_hop_distr.pdf'), C, FS, relative);


% =========================================================================
function render_hop_chart(csvPath, titleText, outFile, C, FS, relative)

    if ~isfile(csvPath)
        fprintf('[WARN] Missing %s — skipping.\n', csvPath);
        return;
    end

    T = readtable(csvPath, 'TextType', 'string');

    % Keep only hop >= 1  (hop=0 are base facts)
    T = T(str2double(string(T.hop)) >= 1, :);

    % Build raw hop counts per method
    MAX_HOP_RAW = 12;
    colOrder = {'baseline', 'synthology'};
    labels   = {'UDM Baseline', 'Synthology'};

    raw = zeros(MAX_HOP_RAW, 2);
    for ci = 1:2
        mask = strcmpi(string(T.method), colOrder{ci});
        sub  = T(mask, :);
        for r = 1:height(sub)
            h = str2double(string(sub.hop(r)));
            if h >= 1 && h <= MAX_HOP_RAW
                raw(h, ci) = str2double(string(sub.count(r)));
            end
        end
    end

    % Binning logic
    binned = zeros(6, 2);
    binned(1, :) = sum(raw(1:3, :), 1);    
    binned(2, :) = raw(4, :);              
    binned(3, :) = raw(5, :);              
    binned(4, :) = raw(6, :);              
    binned(5, :) = raw(7, :);              
    binned(6, :) = sum(raw(8:end, :), 1);  

    % --- RELATIVE DENSITY LOGIC ---
    yLab = 'Positive inferred facts (log)';
    if relative
        % Normalize each column so it sums to 1
        % We use sum(..., 'omitnan') to be safe, though raw starts as zeros
        colSums = sum(binned, 1);
        binned = binned ./ colSums; 
        yLab = 'Relative Density (log)';
    end
    % ------------------------------

    tickLabels = {'1--3', '4', '5', '6', '7', '8+'};
    binned(binned == 0) = NaN;

    % Create figure
    fig = figure('Position', [100, 100, 1000, 500], 'Color', 'w');

    b = bar(1:6, binned, 'grouped');
    b(1).FaceColor = C.KULijsblauw;
    b(2).FaceColor = C.KULcorporate;
    b(1).FaceAlpha = 0.88;
    b(2).FaceAlpha = 0.88;

    set(gca, 'YScale', 'log');
    set(gca, 'XTick', 1:6, 'XTickLabel', tickLabels, 'FontSize', FS);

    xlabel('Proof depth (hops)', 'FontSize', FS, 'FontWeight', 'bold');
    ylabel(yLab,                 'FontSize', FS, 'FontWeight', 'bold');
    title(titleText,             'FontSize', FS + 2, 'FontWeight', 'bold');

    legend(labels, 'Location', 'northeast', 'FontSize', FS - 2, 'Interpreter', 'latex');
    
    box off;
    grid on;
    set(gca, 'GridLineStyle', ':', 'GridAlpha', 0.5);

    % Force delete if file exists to ensure a fresh write
    if isfile(outFile)
        delete(outFile);
    end

    % Save result
    exportgraphics(fig, outFile, 'ContentType', 'vector');
    fprintf('Saved hop chart: %s\n', outFile);
    close(fig);
    
end