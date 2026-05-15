# -*- coding: utf-8 -*-
import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path


class TestParetoDominance:

    def test_dominates_self_is_false(self):
        from prepare import dominates
        a = {"rank_ic": 0.05, "ic_ir": 1.0, "turnover_stability": 0.8}
        assert not dominates(a, a)

    def test_clearly_dominates(self):
        from prepare import dominates
        a = {"rank_ic": 0.05, "ic_ir": 1.5, "turnover_stability": 0.85}
        b = {"rank_ic": 0.03, "ic_ir": 1.0, "turnover_stability": 0.70}
        assert dominates(a, b)

    def test_not_dominates_when_one_metric_worse(self):
        from prepare import dominates
        a = {"rank_ic": 0.05, "ic_ir": 1.5, "turnover_stability": 0.50}
        b = {"rank_ic": 0.03, "ic_ir": 1.0, "turnover_stability": 0.90}
        assert not dominates(a, b)

    def test_not_dominates_when_all_equal(self):
        from prepare import dominates
        a = {"rank_ic": 0.05, "ic_ir": 1.0, "turnover_stability": 0.8}
        b = {"rank_ic": 0.05, "ic_ir": 1.0, "turnover_stability": 0.8}
        assert not dominates(a, b)

    def test_dominates_with_absolute_ic(self):
        from prepare import dominates
        a = {"rank_ic": -0.05, "ic_ir": 1.2, "turnover_stability": 0.75}
        b = {"rank_ic": 0.03, "ic_ir": 1.0, "turnover_stability": 0.70}
        assert dominates(a, b)


class TestParetoArchive:

    def test_pareto_decision_keep_when_frontier_empty(self, tmp_path):
        from prepare import pareto_decision
        archive_path = tmp_path / "test_frontier.json"
        archive_path.write_text(json.dumps({
            "metrics": ["rank_ic", "ic_ir", "turnover_stability"],
            "frontier": [],
            "dominated_count": 0,
            "total_experiments": 0
        }))

        metrics = {"rank_ic": 0.04, "ic_ir": 1.2, "turnover_stability": 0.80}
        status, dominates_list, dominated_by = pareto_decision(
            "test_factor", metrics, str(archive_path))
        assert status == "keep"
        assert len(dominated_by) == 0

    def test_pareto_decision_discard_when_dominated(self, tmp_path):
        from prepare import pareto_decision
        frontier_data = {
            "metrics": ["rank_ic", "ic_ir", "turnover_stability"],
            "frontier": [
                {"name": "strong_factor", "rank_ic": 0.06, "ic_ir": 2.0,
                 "turnover_stability": 0.90}
            ],
            "dominated_count": 5,
            "total_experiments": 10
        }
        archive_path = tmp_path / "frontier.json"
        archive_path.write_text(json.dumps(frontier_data))

        metrics = {"rank_ic": 0.02, "ic_ir": 0.5, "turnover_stability": 0.50}
        status, dominates_list, dominated_by = pareto_decision(
            "weak_factor", metrics, str(archive_path))
        assert status == "discard"
        assert len(dominated_by) > 0

    def test_pareto_decision_keep_non_dominated(self, tmp_path):
        from prepare import pareto_decision
        frontier_data = {
            "metrics": ["rank_ic", "ic_ir", "turnover_stability"],
            "frontier": [
                {"name": "high_ic", "rank_ic": 0.06, "ic_ir": 1.0,
                 "turnover_stability": 0.40},
            ],
            "dominated_count": 3,
            "total_experiments": 8
        }
        archive_path = tmp_path / "frontier.json"
        archive_path.write_text(json.dumps(frontier_data))

        metrics = {"rank_ic": 0.04, "ic_ir": 2.5, "turnover_stability": 0.95}
        status, dominates_list, dominated_by = pareto_decision(
            "stable_factor", metrics, str(archive_path))
        assert status == "keep"
        assert len(dominated_by) == 0

    def test_update_archive_adds_and_removes_dominated(self, tmp_path):
        from prepare import update_archive
        archive_path = tmp_path / "frontier.json"
        archive_path.write_text(json.dumps({
            "metrics": ["rank_ic", "ic_ir", "turnover_stability"],
            "frontier": [
                {"name": "old", "rank_ic": 0.03, "ic_ir": 0.8,
                 "turnover_stability": 0.50}
            ],
            "dominated_count": 0,
            "total_experiments": 5
        }))

        new_factor = {
            "name": "new_better",
            "rank_ic": 0.05,
            "ic_ir": 1.5,
            "turnover_stability": 0.80,
            "description": "better on all metrics",
            "commit": "abc1234",
            "added": "2026-05-15T10:00:00",
            "formula": "ops.cs_rank(close - ops.delay(close, 5))"
        }
        update_archive(new_factor, dominates=["old"], str_path=str(archive_path))

        with open(archive_path) as f:
            updated = json.load(f)
        assert len(updated["frontier"]) == 1
        assert updated["frontier"][0]["name"] == "new_better"
        assert updated["dominated_count"] == 1
        assert updated["total_experiments"] == 6
