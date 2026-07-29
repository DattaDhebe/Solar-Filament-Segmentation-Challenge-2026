import re
import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "sensitive_path",
    [
        "access_token",
        "secrets/kaggle.json",
        "data/raw/example.jpeg",
        "data/external/labels.json",
        "outputs/experiments/run/metrics.json",
        "checkpoints/model.pth",
        "submission.csv",
        "kaggle/kernel-metadata.json",
    ],
)
def test_sensitive_generated_paths_are_ignored(sensitive_path: str) -> None:
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", sensitive_path],
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    assert result.returncode == 0, f"Sensitive path is not ignored: {sensitive_path}"


def test_no_sensitive_or_generated_path_is_tracked() -> None:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    forbidden = re.compile(
        r"(^|/)(access_token[^/]*|kaggle\.json|kernel-metadata\.json)$"
        r"|^data/(raw|interim|processed|external)/"
        r"|^outputs/(?!README\.md$)"
        r"|\.(ckpt|pt|pth|onnx)$"
        r"|(^|/)submission[^/]*\.csv$",
        re.IGNORECASE,
    )
    offenders = [path for path in result.stdout.splitlines() if forbidden.search(path)]
    assert offenders == []
