% predicate_coverage.m
%
% Structural coverage chart: inferred fact count per predicate, comparing
% UDM Baseline vs Synthology.  Predicates where the baseline produces zero
% inferred facts are highlighted — a model trained on that data can never
% learn to predict those predicates.
%
% Reads inferred_predicates_by_method.csv from the pinned report directories.

EXP2_REPORT_DIR = '/dtu/blackhole/16/221590/synthology/reports/experiment_runs/2026-04-17/exp2/report_data/220147_compare/report';
EXP3_REPORT_DIR = '/dtu/blackhole/16/221590/synthology/reports/experiment_runs/2026-04-17/exp3/report_data/162429_compare/report';

% ---------------------------------------------------------------------------

C = kulcolors();

% No LaTeX anywhere — avoids the cmcsc10 font error on HPC.
set(groot, 'defaultTextInterpreter',          'none');
set(groot, 'defaultAxesTickLabelInterpreter', 'none');
set(groot, 'defaultLegendInterpreter',        'none');
set(groot, 'defaultAxesFontSize',   11);
set(groot, 'defaultLegendFontSize', 11);

repoRoot = fileparts(fileparts(mfilename('fullpath')));
outDir   = fullfile(repoRoot, 'paper', 'figures');
if ~exist(outDir, 'dir'), mkdir(outDir); end

render_coverage_chart( ...
    fullfile(EXP2_REPORT_DIR, 'inferred_predicates_by_method.csv'), ...
    'Exp. 2 (Family Tree): Inferred Facts per Predicate', ...
    fullfile(outDir, 'exp2_predicate_coverage.pdf'), C);

render_coverage_chart( ...
    fullfile(EXP3_REPORT_DIR, 'inferred_predicates_by_method.csv'), ...
    'Exp. 3 (OWL2Bench): Inferred Facts per Predicate', ...
    fullfile(outDir, 'exp3_predicate_coverage.pdf'), C);


% =========================================================================
function render_coverage_chart(csvPath, titleText, outFile, C)

    if ~isfile(csvPath)
        fprintf('[WARN] Missing %s — skipping.\n', csvPath);
        return;
    end

    T = readtable(csvPath, 'TextType', 'string');

    % Pivot to one row per predicate.
    predicates = unique(string(T.predicate), 'stable');
    n = numel(predicates);
    b_counts = zeros(n, 1);
    s_counts = zeros(n, 1);

    for i = 1:n
        mb = strcmp(string(T.predicate), predicates(i)) & strcmpi(string(T.method), 'baseline');
        ms = strcmp(string(T.predicate), predicates(i)) & strcmpi(string(T.method), 'synthology');
        if any(mb), b_counts(i) = str2double(string(T.count(mb))); end
        if any(ms), s_counts(i) = str2double(string(T.count(ms))); end
    end

    % Drop predicates where both methods have zero (uninformative).
    keep       = (b_counts > 0) | (s_counts > 0);
    predicates = predicates(keep);
    b_counts   = b_counts(keep);
    s_counts   = s_counts(keep);

    % Sort by Synthology count descending.
    [~, ord]   = sort(s_counts, 'descend');
    predicates = predicates(ord);
    b_counts   = b_counts(ord);
    s_counts   = s_counts(ord);
    n          = numel(predicates);

    % Rows where one method has zero coverage — each gets its own highlight.
    synth_only  = (b_counts == 0) & (s_counts > 0);  % baseline blind spot
    base_only   = (s_counts == 0) & (b_counts > 0);  % synthology blind spot
    n_blind_b   = sum(synth_only);
    n_blind_s   = sum(base_only);

    % NaN so log-scale bars for zero counts simply don't render.
    b_plot = b_counts;  b_plot(b_plot == 0) = NaN;
    s_plot = s_counts;  s_plot(s_plot == 0) = NaN;

    % Figure sizing: tall enough for all rows, wide enough for long names.
    row_px  = 22;
    fig_w   = 920;
    fig_h   = max(400, n * row_px + 120);

    fig = figure('Position', [100, 100, fig_w, fig_h], 'Color', 'w');

    % Explicit axes position — large left margin for predicate names.
    % [left bottom width height] in normalised units.
    left_frac = 220 / fig_w;
    ax = axes('Position', [left_frac, 0.07, 1 - left_frac - 0.04, 0.87]);

    % --- Draw bars first, then set log scale, then overlay patches. -------

    data = [b_plot, s_plot];
    bh = barh(ax, 1:n, data, 'grouped');
    bh(1).FaceColor = C.KULijsblauw;
    bh(2).FaceColor = C.KULcorporate;
    bh(1).FaceAlpha = 0.88;
    bh(2).FaceAlpha = 0.88;

    set(ax, 'XScale', 'log');
    set(ax, 'YTick', 1:n, 'YTickLabel', predicates, 'FontSize', 9);
    set(ax, 'YDir', 'reverse');
    set(ax, 'YLim', [0.5, n + 0.5]);

    % Compute xlim AFTER log scale is active so patch coords are correct.
    xl = xlim(ax);

    % Shade rows where one method has zero coverage.
    % Pink  = baseline blind spot (Synthology covers it, baseline does not).
    % Blue  = Synthology blind spot (baseline covers it, Synthology does not).
    hold(ax, 'on');
    for i = 1:n
        if synth_only(i)
            patch(ax, [xl(1) xl(2) xl(2) xl(1)], [i-0.5 i-0.5 i+0.5 i+0.5], ...
                  [1.00 0.78 0.78], 'EdgeColor', 'none', 'FaceAlpha', 0.28);
        elseif base_only(i)
            patch(ax, [xl(1) xl(2) xl(2) xl(1)], [i-0.5 i-0.5 i+0.5 i+0.5], ...
                  [0.78 0.88 1.00], 'EdgeColor', 'none', 'FaceAlpha', 0.28);
        end
    end
    hold(ax, 'off');

    xlabel(ax, 'Inferred training facts (log scale)', 'FontSize', 11, 'FontWeight', 'bold');
    title(ax,  titleText,                             'FontSize', 11, 'FontWeight', 'bold');

    legend(ax, ...
        {sprintf('UDM Baseline  (%d predicates with zero coverage)', n_blind_b), ...
         sprintf('Synthology  (%d predicates with zero coverage)',   n_blind_s)}, ...
        'Location', 'southeast', 'FontSize', 10, 'Interpreter', 'none');

    box(ax, 'off');
    grid(ax, 'on');
    set(ax, 'GridLineStyle', ':', 'GridAlpha', 0.45, 'XMinorGrid', 'on');

    exportgraphics(fig, outFile, 'ContentType', 'vector');
    fprintf('Saved: %s\n', outFile);
    close(fig);
end
