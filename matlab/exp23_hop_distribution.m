% exp23_hop_distribution.m
%
% Reads the pre-computed hops_by_method.csv produced by the data reporter
% and renders semilogy grouped bar charts for Exp2 and Exp3.
%
% Hops are binned: 1-3 (easy), 4, 5, 6, 7, 8+ (hard).
%
% Pin the exact report directories here — no glob searching, no targets.csv.

EXP2_REPORT_DIR = '/dtu/blackhole/16/221590/synthology/reports/experiment_runs/2026-04-18/exp2/report_data/083304_compare/report';
EXP3_REPORT_DIR = '/dtu/blackhole/16/221590/synthology/reports/experiment_runs/2026-04-18/exp3/report_data/084652_compare/report';

% ---------------------------------------------------------------------------

C = kulcolors();

set(groot, 'defaultTextInterpreter',          'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter',        'latex');
set(groot, 'defaultAxesFontSize',   14);
set(groot, 'defaultLegendFontSize', 12);
set(groot, 'defaultTextFontSize',   12);

repoRoot = fileparts(fileparts(mfilename('fullpath')));
outDir   = fullfile(repoRoot, 'paper', 'figures');
if ~exist(outDir, 'dir'), mkdir(outDir); end

render_hop_chart( ...
    fullfile(EXP2_REPORT_DIR, 'hops_by_method.csv'), ...
    'Experiment 2: Hop Distribution of Positive Inferred Targets (Family Tree)', ...
    fullfile(outDir, 'exp2_hop_distr.pdf'), C);

render_hop_chart( ...
    fullfile(EXP3_REPORT_DIR, 'hops_by_method.csv'), ...
    'Experiment 3: Hop Distribution of Positive Inferred Targets (OWL2Bench)', ...
    fullfile(outDir, 'exp3_hop_distr.pdf'), C);


% =========================================================================
function render_hop_chart(csvPath, titleText, outFile, C)

    if ~isfile(csvPath)
        fprintf('[WARN] Missing %s — skipping.\n', csvPath);
        return;
    end

    T = readtable(csvPath, 'TextType', 'string');

    % Keep only hop >= 1  (hop=0 are base facts)
    T = T(str2double(string(T.hop)) >= 1, :);

    % Build raw hop counts per method (cap at hop 12 for display).
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

    % Bin into: [1-3], [4], [5], [6], [7], [8+].
    binned = zeros(6, 2);
    binned(1, :) = sum(raw(1:3, :), 1);    % easy:  hops 1-3
    binned(2, :) = raw(4, :);              % hop 4
    binned(3, :) = raw(5, :);              % hop 5
    binned(4, :) = raw(6, :);              % hop 6
    binned(5, :) = raw(7, :);              % hop 7
    binned(6, :) = sum(raw(8:end, :), 1);  % hard:  hops 8+

    tickLabels = {'1--3', '4', '5', '6', '7', '8+'};

    % Replace zeros with NaN so log-scale skips missing bars cleanly.
    binned(binned == 0) = NaN;

    fig = figure('Position', [100, 100, 960, 400], 'Color', 'w');

    b = bar(1:6, binned, 'grouped');
    b(1).FaceColor = C.KULijsblauw;
    b(2).FaceColor = C.KULcorporate;
    b(1).FaceAlpha = 0.88;
    b(2).FaceAlpha = 0.88;

    set(gca, 'YScale', 'log');
    set(gca, 'XTick', 1:6, 'XTickLabel', tickLabels, 'FontSize', 12);

    xlabel('Proof depth (hops)',            'FontSize', 13, 'FontWeight', 'bold');
    ylabel('Positive inferred facts (log)', 'FontSize', 13, 'FontWeight', 'bold');
    title(titleText,                        'FontSize', 13, 'FontWeight', 'bold');

    legend(labels, 'Location', 'northeast', 'FontSize', 12, 'Interpreter', 'latex');
    box off;
    grid on;
    set(gca, 'GridLineStyle', ':', 'GridAlpha', 0.5);

    exportgraphics(fig, outFile, 'ContentType', 'vector');
    fprintf('Saved hop chart: %s\n', outFile);
    close(fig);
end
