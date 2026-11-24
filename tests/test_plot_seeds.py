import json
import math
import statistics

from benchmarks import plot_seeds
from benchmarks.plot_seeds import aggregate, build_html, merge_roots, percentile


def test_percentile_interpolates():
    assert percentile([1.0], 0.25) == 1.0
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.5) == 3.0
    assert math.isclose(percentile([0.0, 10.0], 0.25), 2.5)


def test_aggregate_median_and_band_over_seeds():
    runs = {1: [4.0, 2.0, 1.0], 2: [6.0, 3.0, 3.0], 3: [5.0, 5.0, 2.0]}
    a = aggregate(runs)
    assert a["n_runs"] == 3 and a["seeds"] == [1, 2, 3]
    assert a["median"] == [5.0, 3.0, 2.0]
    assert a["lo"][0] <= a["median"][0] <= a["hi"][0]
    assert a["final_median"] == 2.0
    assert a["n_full_budget"] == 3


def test_aggregate_does_not_pad_early_stopped_runs():
    # two runs go to 4 evaluations, one stopped after 2
    runs = {1: [4.0, 3.0, 2.0, 1.0], 2: [5.0, 4.0, 3.0, 2.0], 3: [9.0, 8.0]}
    a = aggregate(runs)
    assert a["n_full_budget"] == 2
    assert a["evals_median"] == 4
    # at eval index 2 only 2 of 3 runs exist (>= half) so it is drawn, using just those two
    assert len(a["median"]) == 4
    assert a["median"][2] == 2.5  # median of [2.0, 3.0], the early-stopped run is not held at 8.0


def test_aggregate_stops_drawing_median_when_fewer_than_half_reached():
    runs = {1: [5.0, 4.0, 3.0, 2.0], 2: [5.0, 4.0], 3: [5.0, 4.0], 4: [5.0, 4.0]}
    a = aggregate(runs)
    assert len(a["median"]) == 2  # only 1 of 4 runs reaches evaluation 3
    assert a["traces"][0] == [5.0, 4.0, 3.0, 2.0]  # but its raw trace is still kept for the faint line


def test_merge_roots_dedupes_vanilla_and_labels_models(tmp_path, monkeypatch):
    def fake_collect(root):
        return {"branin-blind": {"vanilla": {42: [3.0, 2.0]}, "sara-lenz": {42: [3.0, 1.0] if root.name == "a" else [3.0, 0.5]}}}

    monkeypatch.setattr(plot_seeds, "collect_runs", fake_collect)
    a, b = tmp_path / "a", tmp_path / "b"
    single = merge_roots([a])
    assert set(single["branin-blind"]) == {"vanilla", "sara-lenz"}
    both = merge_roots([a, b])
    assert set(both["branin-blind"]) == {"vanilla", "sara-lenz [a]", "sara-lenz [b]"}
    assert both["branin-blind"]["vanilla"] == {42: [3.0, 2.0]}  # from the first root only


def test_build_html_renders_chart_table_and_summary(tmp_path):
    merged = {"hartmann6-blind": {
        "vanilla": {42: [2.0, 1.0, 0.1], 43: [2.5, 1.5, 0.2]},
        "sara-lenz": {42: [2.0, 1.8, 1.7], 43: [2.5, 2.4]},
    }}
    page, summary = build_html(merged, "t", fig_dir=tmp_path / "figs")
    assert (tmp_path / "figs" / "hartmann6-blind.png").stat().st_size > 1000
    assert "<img" in page and "data:image/png;base64," in page and "hartmann6-blind" in page
    assert "<td>sara-lenz</td><td>2</td><td>1/2</td>" in page  # 2 runs, 1 used the full budget
    assert summary["hartmann6-blind"]["sara-lenz"]["n_full_budget"] == 1
    json.dumps(summary)  # serialisable
    empty_page, empty_summary = build_html({}, "t")
    assert "No scoreable runs" in empty_page and empty_summary == {}


def test_aggregate_mean_and_standard_error_carry_early_stops_forward():
    runs = {1: [4.0, 2.0], 2: [6.0, 4.0, 2.0], 3: [5.0, 3.0, 1.0]}  # run 1 stopped after 2 evaluations
    a = aggregate(runs)
    # at index 2 run 1 is carried forward at its last value 2.0
    assert math.isclose(a["mean"][2], (2.0 + 2.0 + 1.0) / 3)
    assert math.isclose(a["se"][2], statistics.stdev([2.0, 2.0, 1.0]) / math.sqrt(3))
    assert a["final_mean"] == a["mean"][-1] and a["evals_median"] == 3
    single = aggregate({1: [3.0, 2.0]})
    assert single["se"] == [0.0, 0.0]  # one seed: no uncertainty estimate


def test_mean_and_median_figures_render_with_median_stop_marker(tmp_path):
    merged = {"h-blind": {
        "vanilla": {s: [3.0 - 0.1 * i - 0.05 * s for i in range(10)] for s in (1, 2, 3)},
        "agent": {s: [3.0 - 0.1 * i for i in range(4 + s)] for s in (1, 2, 3)},
    }}
    for stat in ("mean", "median"):
        page, summary = build_html(merged, "t", fig_dir=tmp_path / stat, stat=stat)
        assert (tmp_path / stat / "h-blind.png").stat().st_size > 1000
        assert "&plusmn;" in page and summary["h-blind"]["agent"]["final_se"] >= 0
    assert "carried forward" in build_html(merged, "t", stat="mean")[0]


def test_merge_roots_display_names(tmp_path, monkeypatch):
    monkeypatch.setattr(plot_seeds, "collect_runs", lambda root: {"g": {"vanilla": {1: [1.0]}, "sara-lenz": {1: [2.0]}}})
    out = merge_roots([tmp_path / "fix", tmp_path / "nofix"],
                      {"fix": "Agent, forced to continue", "nofix": "Agent, first stop", "vanilla": "Vanilla BO"})
    assert set(out["g"]) == {"Vanilla BO", "Agent, forced to continue", "Agent, first stop"}
