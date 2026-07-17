from pathlib import Path


def test_python_ci_installs_a_non_vulnerable_setuptools_release() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'setuptools>=83' in pyproject
    assert '"setuptools>=83"' in workflow


def test_frontend_ci_installs_pnpm_before_setup_node_uses_its_cache() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    pnpm_setup = workflow.index("pnpm/action-setup@")
    node_setup = workflow.index("actions/setup-node@")
    cache_config = workflow.index("cache: pnpm")

    assert pnpm_setup < node_setup < cache_config
