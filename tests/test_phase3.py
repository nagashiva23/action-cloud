"""
Unit tests for ActionCloud Phase 3 features:
  - Multi-provider LLM factory (MockLLM, GeminiLLM, GroqLLM, AnthropicLLM)
  - Benchmark task dataset loader & validation
  - MetricCalculator aggregation logic
"""

from __future__ import annotations

import os
import pytest

from actioncloud.llm import GeminiLLM, GroqLLM, MockLLM, get_llm
from actioncloud.metrics import MetricCalculator


class TestMultiProviderLLMFactory:
    def test_default_mock_provider(self):
        llm = get_llm("mock")
        assert isinstance(llm, MockLLM)
        assert llm.model == "mock-llm-v1"

    def test_gemini_provider_instantiation(self):
        os.environ["GEMINI_API_KEY"] = "fake-key-for-test"
        llm = get_llm("gemini")
        assert isinstance(llm, GeminiLLM)
        assert llm.model == "gemini-1.5-flash"

    def test_groq_provider_instantiation(self):
        os.environ["GROQ_API_KEY"] = "fake-key-for-test"
        llm = get_llm("groq")
        assert isinstance(llm, GroqLLM)
        assert llm.model == "llama-3.3-70b-versatile"

    def test_invalid_provider_raises(self):
        with pytest.raises(ValueError):
            get_llm("invalid-provider-name")


class TestBenchmarkDataset:
    def test_benchmark_tasks_file_exists_and_valid(self):
        import json
        from pathlib import Path

        tasks_path = Path(__file__).parents[1] / "src" / "actioncloud" / "benchmark" / "tasks.json"
        assert tasks_path.exists()
        with open(tasks_path, "r", encoding="utf-8") as f:
            tasks = json.load(f)

        assert len(tasks) >= 20
        assert len(tasks) >= 20
        for t in tasks:
            assert "role" in t
            assert "task" in t
            assert "task_key" in t
            assert "technologies" in t


class TestMetricCalculator:
    def test_calculate_run_metrics_structure(self):
        metrics = MetricCalculator.calculate_run_metrics(run_id="nonexistent-run-id")
        assert "systems" in metrics
        assert "comparative_metrics" in metrics
        assert "governance_tier_distribution" in metrics
