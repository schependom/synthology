% ontology_hop_theory.m
%
% Overlays the theoretical hop-depth distribution (computed by
% scripts/analyze_ontology_hops.py) with the actual Synthology-generated
% distribution from a targets.csv file.
%
% The theoretical curve confirms that the hop distribution observed in the
% generated data reflects the inference structure of the ontology itself,
% not a limitation of the generation method.
%
% Usage (interactive):
%   % Set the two paths and call the script, or use the helper at the bottom:
%   plot_hop_theory( ...
%       'reports/ontology_hop_analysis/owl2bench/theory_hop_distribution.csv', ...
%       'data/exp3/synthology/owl2bench_20/train/targets.csv', ...
%       'OWL2Bench', ...
%       'paper/figures/exp3_hop_theory.pdf');
%
%   plot_hop_theory( ...
%       'reports/ontology_hop_analysis/family/theory_hop_distribution.csv', ...
%       'data/exp2/synthology/family_tree/train/targets.csv', ...
%       'Family Tree', ...
%       'paper/figures/exp2_hop_theory.pdf');

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

% ---- Exp 2 (Family Tree) -----------------------------------------------
plot_hop_theory( ...
    fullfile(repoRoot, 'reports', 'ontology_hop_analysis', 'family', 'theory_hop_distribution.csv'), ...
    fullfile(repoRoot, 'data', 'exp2', 'synthology', 'family_tree', 'train', 'targets.csv'), ...
    'Family Tree', ...
    fullfile(outDir, 'exp2_hop_theory.pdf'), ...
    C);

% ---- Exp 3 (OWL2Bench) -------------------------------------------------
% Auto-detect the latest balanced OWL2Bench synthology run.
owl2BenchCandidates = dir(fullfile(repoRoot, 'data', 'exp3', 'balanced', 'owl2bench_*', 'train', 'targets.csv'));
if isempty(owl2BenchCandidates)
    % Fall back to unbalanced data if balancing hasn't been run yet.
    owl2BenchCandidates = dir(fullfile(repoRoot, 'data', 'exp3', 'synthology', 'owl2bench_*', 'train', 'targets.csv'));
end

if ~isempty(owl2BenchCandidates)
    [~, idx] = max([owl2BenchCandidates.datenum]);
    owl2BenchTargets = fullfile(owl2BenchCandidates(idx).folder, owl2BenchCandidates(idx).name);
    plot_hop_theory( ...
        fullfile(repoRoot, 'reports', 'ontology_hop_analysis', 'owl2bench', 'theory_hop_distribution.csv'), ...
        owl2BenchTargets, ...
        'OWL2Bench', ...
        fullfile(outDir, 'exp3_hop_theory.pdf'), ...
        C);
else
    fprintf('[WARN] No OWL2Bench Synthology targets found. Skipping exp3_hop_theory.pdf\n');
end


% =========================================================================
function plot_hop_theory(theoryCsvPath, targetsCsvPath, ontologyName, outFile, C)
% PLOT_HOP_THEORY  Overlay theoretical vs actual hop distribution.
%
%   theoryCsvPath  : path to theory_hop_distribution.csv
%   targetsCsvPath : path to train/targets.csv from the Synthology run
%   ontologyName   : short label for the title
%   outFile        : output PDF path
%   C              : KU Leuven colour struct

    if ~isfile(theoryCsvPath)
        fprintf('[WARN] Theory CSV not found: %s\n  Run scripts/analyze_ontology_hops.py first.\n', theoryCsvPath);
        return;
    end
    if ~isfile(targetsCsvPath)
        fprintf('[WARN] Targets CSV not found: %s\n', targetsCsvPath);
        return;
    end

    % --- Load theoretical distribution ----------------------------------
    theoryT = readtable(theoryCsvPath, 'TextType', 'string');
    theoryDepths = double(theoryT.depth);
    theoryFreq   = double(theoryT.relative_frequency);

    % --- Load actual inferred-only hop counts from targets.csv ----------
    T = readtable(targetsCsvPath, 'TextType', 'string');

    hopsNum = numeric_col(T.hops);
    types   = lower(string(T.type));
    labels  = numeric_col(T.label);

    inferredMask = startsWith(types, 'inf');
    posMask      = labels == 1;
    keep         = inferredMask & posMask & ~isnan(hopsNum) & (hopsNum >= 1);
    actualHops   = hopsNum(keep);

    % Normalise actual distribution to relative frequency.
    maxD = max([theoryDepths; actualHops]);
    actualCounts = arrayfun(@(d) sum(actualHops == d), 1:maxD);
    actualFreq   = actualCounts / max(sum(actualCounts), 1);

    % --- Plot -----------------------------------------------------------
    fig = figure('Position', [100, 100, 1000, 380]);

    allDepths = 1:maxD;

    % Theoretical: step/bar in background.
    theoryAll = zeros(1, maxD);
    theoryAll(theoryDepths) = theoryFreq;
    b = bar(allDepths, theoryAll, 1.0, 'FaceColor', C.KULijsblauw, ...
            'FaceAlpha', 0.35, 'EdgeColor', 'none');

    hold on;

    % Actual: solid line with markers.
    plot(allDepths, actualFreq, '-o', ...
         'Color', C.KULcorporate, ...
         'LineWidth', 2, ...
         'MarkerSize', 6, ...
         'MarkerFaceColor', C.KULcorporate);

    hold off;

    set(gca, 'XTick', allDepths, 'FontSize', 12);
    xlabel('Proof depth (hops)', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel('Relative frequency', 'FontSize', 13, 'FontWeight', 'bold');
    title(sprintf('%s: Theoretical vs.\ Actual Hop Distribution (Synthology)', ontologyName), ...
          'FontSize', 13, 'FontWeight', 'bold');

    legend({'Theoretical (rule-graph)', 'Synthology (actual)'}, ...
           'Location', 'northeast', 'FontSize', 11, 'Interpreter', 'latex');
    box off;
    grid on;
    set(gca, 'GridLineStyle', ':', 'GridAlpha', 0.5);

    exportgraphics(fig, outFile, 'ContentType', 'vector');
    fprintf('Saved: %s\n', outFile);
    close(fig);
end


function nums = numeric_col(col)
    if isnumeric(col)
        nums = double(col);
        return;
    end
    nums = str2double(string(col));
end
