% exp23_hop_distribution.m
%
% Reads the pre-computed hops_by_method.csv produced by the data reporter
% and renders semilogy line charts for Exp2 and Exp3.
%
% Pin the exact report directories here — no glob searching, no targets.csv.

EXP2_REPORT_DIR = '/dtu/blackhole/16/221590/synthology/reports/experiment_runs/2026-04-17/exp2/report_data/220147_compare/report';
EXP3_REPORT_DIR = '/dtu/blackhole/16/221590/synthology/reports/experiment_runs/2026-04-17/exp3/report_data/162429_compare/report';

% ---------------------------------------------------------------------------

C = kulcolors();

% No LaTeX — avoids cmcsc10 font error on HPC.
set(groot, 'defaultTextInterpreter',          'none');
set(groot, 'defaultAxesTickLabelInterpreter', 'none');
set(groot, 'defaultLegendInterpreter',        'none');
set(groot, 'defaultAxesFontSize',   14);
set(groot, 'defaultLegendFontSize', 12);
set(groot, 'defaultTextFontSize',   12);

repoRoot = fileparts(fileparts(mfilename('fullpath')));
outDir   = fullfile(repoRoot, 'paper', 'figures');
if ~exist(outDir, 'dir'), mkdir(outDir); end

render_hop_chart( ...
    fullfile(EXP2_REPORT_DIR, 'hops_by_method.csv'), ...
    'Exp. 2 (Family Tree): Hop Distribution of Positive Inferred Targets', ...
    fullfile(outDir, 'exp2_hop_distr.pdf'), C);

render_hop_chart( ...
    fullfile(EXP3_REPORT_DIR, 'hops_by_method.csv'), ...
    'Exp. 3 (OWL2Bench): Hop Distribution of Positive Inferred Targets', ...
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

    maxHop   = max(str2double(string(T.hop)));
    hopTicks = 1:maxHop;

    % Build count matrix: rows = hops, cols = [baseline, synthology]
    colOrder = {'baseline', 'synthology'};
    counts   = zeros(maxHop, 2);
    for ci = 1:2
        mask = strcmpi(string(T.method), colOrder{ci});
        sub  = T(mask, :);
        for r = 1:height(sub)
            h = str2double(string(sub.hop(r)));
            if h >= 1 && h <= maxHop
                counts(h, ci) = str2double(string(sub.count(r)));
            end
        end
    end

    fig = figure('Position', [100, 100, 960, 380], 'Color', 'w');

    styles = {'-o', '-s'};
    colors = {C.KULijsblauw, C.KULcorporate};
    labels = {'UDM Baseline', 'Synthology'};

    hold on;
    ph = gobjects(2, 1);
    for ci = 1:2
        y     = counts(:, ci);
        valid = y > 0;
        if any(valid)
            ph(ci) = semilogy(hopTicks(valid), y(valid), styles{ci}, ...
                'Color',           colors{ci}, ...
                'LineWidth',       2.2, ...
                'MarkerSize',      8, ...
                'MarkerFaceColor', colors{ci});
        end
    end
    hold off;

    set(gca, 'XTick', hopTicks, 'FontSize', 12);
    xlabel('Proof depth (hops)',            'FontSize', 13, 'FontWeight', 'bold');
    ylabel('Positive inferred facts (log)', 'FontSize', 13, 'FontWeight', 'bold');
    title(titleText,                        'FontSize', 13, 'FontWeight', 'bold');

    % Only include handles that were actually plotted.
    valid_ph  = ph(isgraphics(ph));
    valid_lbl = labels(isgraphics(ph));
    legend(valid_ph, valid_lbl, 'Location', 'northeast', 'FontSize', 12, 'Interpreter', 'none');

    box off;
    grid on;
    set(gca, 'GridLineStyle', ':', 'GridAlpha', 0.5);

    exportgraphics(fig, outFile, 'ContentType', 'vector');
    fprintf('Saved hop chart: %s\n', outFile);
    close(fig);
end
