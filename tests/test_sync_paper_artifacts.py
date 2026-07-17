from __future__ import annotations

import json

from scripts import sync_paper_artifacts


def test_sync_and_check_paper_artifact(tmp_path, monkeypatch) -> None:
    repository = tmp_path / "repository"
    experiments = tmp_path / "experiments"
    source = experiments / "run-v1" / "figures" / "figure.txt"
    source.parent.mkdir(parents=True)
    source.write_text("measured-data\n", encoding="utf-8")
    repository.mkdir()

    monkeypatch.setattr(sync_paper_artifacts, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(
        sync_paper_artifacts,
        "LOCK_PATH",
        repository / "paper" / "artifact_checksums.json",
    )
    groups = [
        {
            "id": "test-v1",
            "generator": "generator.py",
            "source_root": "run-v1",
            "files": [
                {
                    "source": "figures/figure.txt",
                    "destination": "paper/figures/test/figure.txt",
                }
            ],
        }
    ]

    records = sync_paper_artifacts._records(groups, experiments)
    assert sync_paper_artifacts._sync(records, records) == 0
    assert sync_paper_artifacts._check(records) == 0
    assert (
        repository / "paper" / "figures" / "test" / "figure.txt"
    ).read_text(encoding="utf-8") == "measured-data\n"

    lock = json.loads(
        (repository / "paper" / "artifact_checksums.json").read_text(
            encoding="utf-8"
        )
    )
    assert lock["artifacts"][0]["destination"] == (
        "paper/figures/test/figure.txt"
    )
