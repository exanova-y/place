#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import subprocess
import tomllib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = REPOSITORY / "trees" / "eeg-inventory-datasheet.tree"
SNAPSHOT_DATE = "2026-07-13"
LATEST_EVIDENCE = "2026-06-28"
COLOR_VERIFIED_DATE = "2026-07-14"
COLOR_REPOSITORY = "https://github.com/bmorphism/Gay.jl"
COLOR_DEFAULT_BRANCH = "gay"
EXPECTED_RAW_UNITS = 8
EXPECTED_RAW_FILES = 28
EXPECTED_REAL_TUI_UNITS = 11
EXPECTED_REAL_TUI_FILES = 18
EXPECTED_SIM_TUI_UNITS = 6
EXPECTED_SIM_TUI_FILES = 11
EXPECTED_COMPANION_FILES = {
    "face": 6,
    "screen": 21,
    "ancillary": 4,
}
EXPECTED_COMPANION_CLASS_FILES = {
    "OFR-A01": 6,
    "OFR-A02": 7,
    "OFR-A03": 4,
    "OFR-B01": 10,
    "OFR-B02": 4,
}
EXPECTED_TOTAL_FILES = 88

RAW_KIND_SPECS = {
    ".bin": ("raw-packet-stream", "BIN", 0),
    ".json": ("run-metadata", "JSON", 1),
    ".decode.json": ("decode-metadata", "JSON", 2),
    ".events.jsonl": ("derived-event-stream", "JSONL", 3),
    ".npz": ("decoded-array", "NPZ", 4),
}
TUI_KIND_SPECS = {
    "events.jsonl": ("derived-event-stream", "JSONL", 0),
    "status.json": ("status-rollup", "JSON", 1),
}
COMPANION_KIND_SPECS = {
    "face": {
        ".mov": ("face-video", "MOV"),
    },
    "screen": {
        ".mov": ("screen-video", "MOV"),
        ".m4a": ("screen-audio", "M4A"),
    },
    "ancillary": {
        ".md": ("supporting-document", "Markdown"),
        ".pdf": ("supporting-document", "PDF"),
    },
}
SIMULATE_RE = re.compile(rb'"simulate"\s*:\s*(true|false)')
COLOR_PROGRAM = r"""
using Gay, Printf

used = Set{String}()
for line in eachline(stdin)
    alias, target_text = split(line, '\t')
    target = parse(Int, target_text)
    seed = Gay.stable_seed(alias)
    matched = false
    for index in 0:65535
        color = Gay.color_at(index; seed=seed)
        actual = Int(Gay.trit(index; seed=seed))
        if actual == target && !(color in used)
            push!(used, color)
            @printf("%s\t%d\t%s\t%d\t%016X\n", alias, index, color, actual, seed)
            matched = true
            break
        end
    end
    matched || error("no unused trit-matched color for " * alias)
end
"""


@dataclass
class Artifact:
    class_code: str
    unit_alias: str
    kind: str
    format: str
    size: int
    source: Path
    artifact_alias: str = ""
    trit: int = 0
    seed: int = 0
    color_index: int = 0
    color: str = ""


@dataclass(frozen=True)
class ColorProvenance:
    repository: str
    default_branch: str
    revision: str
    git_tree: str
    describe: str
    project_uuid: str
    project_version: str
    julia_version: str
    verified_date: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the privacy-safe public inventory from local file surfaces. "
            "Private filenames, paths, timestamps, and hashes are used only for local "
            "validation and never written to the public tree."
        )
    )
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--raw-manifest", required=True, type=Path)
    parser.add_argument("--tui-runs-dir", required=True, type=Path)
    parser.add_argument("--companion-root", required=True, type=Path)
    parser.add_argument("--color-project", required=True, type=Path)
    parser.add_argument("--private-registry", required=True, type=Path)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the target differs from freshly generated output",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_raw_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = {"sha256", "bytes", "relative_path"}
        if set(reader.fieldnames or ()) != expected:
            raise RuntimeError(f"raw manifest columns must be {sorted(expected)}")
        rows = list(reader)
    if len(rows) != EXPECTED_RAW_UNITS:
        raise RuntimeError(
            f"expected {EXPECTED_RAW_UNITS} canonical raw rows, found {len(rows)}"
        )
    return rows


def raw_kind(path: Path) -> tuple[str, str, str, int]:
    for suffix in sorted(RAW_KIND_SPECS, key=len, reverse=True):
        if path.name.endswith(suffix):
            kind, format_name, order = RAW_KIND_SPECS[suffix]
            return path.name[: -len(suffix)], kind, format_name, order
    raise RuntimeError(f"unrecognized raw companion type: {path.name}")


def collect_raw_artifacts(raw_dir: Path, manifest_path: Path) -> list[Artifact]:
    manifest = parse_raw_manifest(manifest_path)
    groups: dict[str, list[tuple[int, Path, str, str]]] = defaultdict(list)
    files = sorted(path for path in raw_dir.iterdir() if path.is_file())
    if len(files) != EXPECTED_RAW_FILES:
        raise RuntimeError(
            f"expected {EXPECTED_RAW_FILES} raw-class files, found {len(files)}"
        )
    for path in files:
        stem, kind, format_name, order = raw_kind(path)
        groups[stem].append((order, path, kind, format_name))

    ordered_stems: list[str] = []
    for row in manifest:
        relative_path = Path(row["relative_path"])
        raw_path = raw_dir / relative_path.name
        if not raw_path.is_file():
            raise RuntimeError("canonical raw manifest references a missing file")
        expected_size = int(row["bytes"])
        if raw_path.stat().st_size != expected_size:
            raise RuntimeError("canonical raw file size does not match its manifest")
        if sha256_file(raw_path) != row["sha256"]:
            raise RuntimeError("canonical raw file hash does not match its manifest")
        stem, kind, _format_name, _order = raw_kind(raw_path)
        if kind != "raw-packet-stream":
            raise RuntimeError("canonical raw manifest must contain only BIN files")
        if stem in ordered_stems:
            raise RuntimeError("canonical raw manifest contains a duplicate unit")
        ordered_stems.append(stem)

    if set(groups) != set(ordered_stems):
        raise RuntimeError("raw companion groups differ from the canonical manifest")

    artifacts: list[Artifact] = []
    for stem in ordered_stems:
        unit_alias = "OFR-C01-U01"
        for _order, path, kind, format_name in sorted(groups[stem]):
            artifacts.append(
                Artifact(
                    class_code="OFR-C01",
                    unit_alias=unit_alias,
                    kind=kind,
                    format=format_name,
                    size=path.stat().st_size,
                    source=path,
                )
            )
    return artifacts


def simulation_state(files: list[Path]) -> bool:
    states: set[bool] = set()
    for path in files:
        with path.open("rb") as handle:
            prefix = handle.read(64 * 1024)
        states.update(match == b"true" for match in SIMULATE_RE.findall(prefix))
    if len(states) != 1:
        raise RuntimeError("TUI run must expose one unambiguous simulate state")
    return states.pop()


def collect_tui_artifacts(tui_runs_dir: Path) -> tuple[list[Artifact], list[Artifact]]:
    runs: list[tuple[Path, list[Path], bool]] = []
    for run_dir in sorted(path for path in tui_runs_dir.iterdir() if path.is_dir()):
        files = sorted(path for path in run_dir.iterdir() if path.is_file())
        if not files or "events.jsonl" not in {path.name for path in files}:
            raise RuntimeError("every TUI run must contain events.jsonl")
        unknown = {path.name for path in files} - set(TUI_KIND_SPECS)
        if unknown:
            raise RuntimeError("unrecognized TUI artifact type")
        runs.append((run_dir, files, simulation_state(files)))

    real_runs = [run for run in runs if not run[2]]
    simulated_runs = [run for run in runs if run[2]]
    if len(real_runs) != EXPECTED_REAL_TUI_UNITS:
        raise RuntimeError(
            f"expected {EXPECTED_REAL_TUI_UNITS} real TUI runs, found {len(real_runs)}"
        )
    if len(simulated_runs) != EXPECTED_SIM_TUI_UNITS:
        raise RuntimeError(
            f"expected {EXPECTED_SIM_TUI_UNITS} simulated TUI runs, "
            f"found {len(simulated_runs)}"
        )

    def make_artifacts(
        selected_runs: list[tuple[Path, list[Path], bool]],
        class_code: str,
        unit_prefix: str,
        single_unit_alias: str | None = None,
    ) -> list[Artifact]:
        result: list[Artifact] = []
        for unit_index, (_run_dir, files, _simulated) in enumerate(
            selected_runs, start=1
        ):
            unit_alias = single_unit_alias or f"{unit_prefix}{unit_index:02d}"
            for path in sorted(files, key=lambda item: TUI_KIND_SPECS[item.name][2]):
                kind, format_name, _order = TUI_KIND_SPECS[path.name]
                result.append(
                    Artifact(
                        class_code=class_code,
                        unit_alias=unit_alias,
                        kind=kind,
                        format=format_name,
                        size=path.stat().st_size,
                        source=path,
                    )
                )
        return result

    real = make_artifacts(
        real_runs,
        "OFR-C01",
        "",
        "OFR-C01-U01",
    )
    simulated = make_artifacts(
        simulated_runs,
        "OFR-C02",
        "OFR-C02-U",
    )
    if len(real) != EXPECTED_REAL_TUI_FILES:
        raise RuntimeError(
            f"expected {EXPECTED_REAL_TUI_FILES} real TUI files, found {len(real)}"
        )
    if len(simulated) != EXPECTED_SIM_TUI_FILES:
        raise RuntimeError(
            f"expected {EXPECTED_SIM_TUI_FILES} simulated TUI files, "
            f"found {len(simulated)}"
        )
    return real, simulated


def collect_companion_artifacts(companion_root: Path) -> list[Artifact]:
    classified: dict[str, list[tuple[Path, str, str]]] = defaultdict(list)
    for directory_name in ("face", "screen", "ancillary"):
        directory = companion_root / directory_name
        if not directory.is_dir():
            raise RuntimeError(f"missing companion directory: {directory_name}")
        files = sorted(path for path in directory.iterdir() if path.is_file())
        expected_count = EXPECTED_COMPANION_FILES[directory_name]
        if len(files) != expected_count:
            raise RuntimeError(
                f"expected {expected_count} {directory_name} files, found {len(files)}"
            )
        kind_specs = COMPANION_KIND_SPECS[directory_name]
        for path in files:
            suffix = path.suffix.lower()
            if suffix not in kind_specs:
                raise RuntimeError(
                    f"unrecognized {directory_name} companion format: {suffix}"
                )
            kind, format_name = kind_specs[suffix]
            if directory_name == "face":
                class_code = "OFR-A01"
            elif directory_name == "ancillary":
                class_code = "OFR-A03"
            elif "2026-03-" in path.name:
                class_code = "OFR-B01"
            elif "2026-04-" in path.name:
                class_code = "OFR-A02"
            else:
                class_code = "OFR-B02"
            classified[class_code].append((path, kind, format_name))

    actual_counts = {
        class_code: len(entries) for class_code, entries in classified.items()
    }
    if actual_counts != EXPECTED_COMPANION_CLASS_FILES:
        raise RuntimeError(
            "companion cohort split differs from the approved inventory: "
            f"expected {EXPECTED_COMPANION_CLASS_FILES}, found {actual_counts}"
        )

    artifacts: list[Artifact] = []
    for class_code in EXPECTED_COMPANION_CLASS_FILES:
        unit_prefix = f"{class_code}-U"
        for unit_index, (path, kind, format_name) in enumerate(
            classified[class_code], start=1
        ):
            artifacts.append(
                Artifact(
                    class_code=class_code,
                    unit_alias=f"{unit_prefix}{unit_index:02d}",
                    kind=kind,
                    format=format_name,
                    size=path.stat().st_size,
                    source=path,
                )
            )
    return artifacts


def assign_aliases_and_trits(artifacts: list[Artifact]) -> None:
    unit_positions: Counter[str] = Counter()
    balanced_prefix = len(artifacts) - (len(artifacts) % 3)
    for artifact_index, artifact in enumerate(artifacts, start=1):
        unit_positions[artifact.unit_alias] += 1
        artifact.artifact_alias = (
            f"{artifact.unit_alias}-F{unit_positions[artifact.unit_alias]:02d}"
        )
        artifact.trit = (
            (-1, 0, 1)[(artifact_index - 1) % 3]
            if artifact_index <= balanced_prefix
            else 0
        )
    if len(artifacts) != EXPECTED_TOTAL_FILES:
        raise RuntimeError(
            f"expected {EXPECTED_TOTAL_FILES} total files, found {len(artifacts)}"
        )
    if sum(artifact.trit for artifact in artifacts) % 3 != 0:
        raise RuntimeError("GF(3) conservation failed")


def git_output(project: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(project), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def color_provenance(project: Path) -> ColorProvenance:
    if git_output(project, "status", "--porcelain"):
        raise RuntimeError("color project must be a clean checkout")
    revision = git_output(project, "rev-parse", "HEAD")
    git_tree = git_output(project, "rev-parse", "HEAD^{tree}")
    describe = git_output(project, "describe", "--tags", "--always", "--dirty")
    upstream = subprocess.run(
        [
            "git",
            "ls-remote",
            COLOR_REPOSITORY,
            f"refs/heads/{COLOR_DEFAULT_BRANCH}",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not upstream:
        raise RuntimeError("canonical color upstream branch was not found")
    upstream_revision = upstream.split()[0]
    if revision != upstream_revision:
        raise RuntimeError(
            "color checkout is not the live canonical upstream head: "
            f"local {revision}, upstream {upstream_revision}"
        )
    with (project / "Project.toml").open("rb") as handle:
        project_data = tomllib.load(handle)
    project_uuid = str(project_data["uuid"])
    project_version = str(project_data["version"])
    julia_output = subprocess.run(
        ["julia", "--startup-file=no", "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    julia_version = julia_output.removeprefix("julia version ")
    with (project / "Manifest.toml").open("rb") as handle:
        manifest_data = tomllib.load(handle)
    manifest_julia_version = str(manifest_data["julia_version"])
    if julia_version != manifest_julia_version:
        raise RuntimeError(
            "runtime does not match the color project manifest: "
            f"runtime {julia_version}, manifest {manifest_julia_version}"
        )
    return ColorProvenance(
        repository=COLOR_REPOSITORY,
        default_branch=COLOR_DEFAULT_BRANCH,
        revision=revision,
        git_tree=git_tree,
        describe=describe,
        project_uuid=project_uuid,
        project_version=project_version,
        julia_version=julia_version,
        verified_date=COLOR_VERIFIED_DATE,
    )


def assign_colors(artifacts: list[Artifact], project: Path) -> None:
    color_input = "".join(
        f"{artifact.artifact_alias}\t{artifact.trit}\n"
        for artifact in artifacts
    )
    environment = os.environ.copy()
    environment["JULIA_NUM_THREADS"] = "1"
    result = subprocess.run(
        [
            "julia",
            "--startup-file=no",
            f"--project={project}",
            "-e",
            COLOR_PROGRAM,
        ],
        input=color_input,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    output_by_alias: dict[str, tuple[int, str, int, int]] = {}
    for line in result.stdout.splitlines():
        alias, index_text, color, trit_text, seed_text = line.split("\t")
        output_by_alias[alias] = (
            int(index_text),
            color,
            int(trit_text),
            int(seed_text, 16),
        )
    if set(output_by_alias) != {
        artifact.artifact_alias for artifact in artifacts
    }:
        raise RuntimeError("color output does not cover every artifact alias")
    if len({values[1] for values in output_by_alias.values()}) != len(artifacts):
        raise RuntimeError("artifact colors must be unique")

    for artifact in artifacts:
        color_index, color, trit, seed = output_by_alias[artifact.artifact_alias]
        if trit != artifact.trit:
            raise RuntimeError(
                f"color-kernel trit mismatch for {artifact.artifact_alias}: "
                f"expected {artifact.trit}, received {trit}"
            )
        artifact.color_index = color_index
        artifact.color = color
        artifact.seed = seed


def private_registry_document(artifacts: list[Artifact]) -> str:
    rows = [
        "artifact_alias\tclass_code\tunit_alias\tkind\tformat\tbytes\tsha256\tsource"
    ]
    for artifact in artifacts:
        values = (
            artifact.artifact_alias,
            artifact.class_code,
            artifact.unit_alias,
            artifact.kind,
            artifact.format,
            str(artifact.size),
            sha256_file(artifact.source),
            str(artifact.source),
        )
        if any("\t" in value or "\n" in value for value in values):
            raise RuntimeError("private registry value contains a TSV delimiter")
        rows.append("\t".join(values))
    return "\n".join(rows) + "\n"


def sync_private_registry(path: Path, document: str, check: bool) -> None:
    if check:
        if not path.is_file() or path.read_text(encoding="utf-8") != document:
            raise RuntimeError("restricted private alias registry is stale")
        if path.stat().st_mode & 0o077:
            raise RuntimeError("restricted private alias registry must have mode 0600")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, path)


def forest_lines(
    artifacts: list[Artifact],
    provenance: ColorProvenance,
) -> list[str]:
    total_bytes = sum(artifact.size for artifact in artifacts)
    requestable = [
        artifact for artifact in artifacts if artifact.class_code != "OFR-C02"
    ]
    tooling = [
        artifact
        for artifact in artifacts
        if artifact.class_code == "OFR-C02"
    ]
    unit_count = len({artifact.unit_alias for artifact in artifacts})
    trit_counts = Counter(artifact.trit for artifact in artifacts)

    lines = [
        r"\title{Public EEG File Inventory and Deterministic Color Ledger}",
        r"\taxon{inventory}",
        r"\author{monaduck1069}",
        rf"\date{{{SNAPSHOT_DATE}}}",
        r"\tag{bci}",
        r"\tag{eeg}",
        r"\tag{data-request}",
        r"\tag{cohort}",
        r"\tag{color-ledger}",
        r"\meta{place}{place://public/inventory}",
        r"\meta{source}{private://inventory/source}",
        r"\meta{request}{private contact route on parent page}",
        r"\meta{access}{request-only; case-by-case review}",
        r"\meta{status}{public alias ledger; no file-level release authorization}",
        r"\meta{private-registry}{maintainer-held; not published}",
        rf"\meta{{snapshot-date}}{{{SNAPSHOT_DATE}}}",
        rf"\meta{{source-latest-evidence}}{{{LATEST_EVIDENCE}}}",
        rf"\meta{{available-units}}{{{unit_count}}}",
        rf"\meta{{available-files}}{{{len(artifacts)}}}",
        rf"\meta{{available-bytes}}{{{total_bytes}}}",
        rf"\meta{{requestable-files}}{{{len(requestable)}}}",
        rf"\meta{{tooling-fixture-files}}{{{len(tooling)}}}",
        rf"\meta{{trit-minus}}{{{trit_counts[-1]}}}",
        rf"\meta{{trit-zero}}{{{trit_counts[0]}}}",
        rf"\meta{{trit-plus}}{{{trit_counts[1]}}}",
        rf"\meta{{trit-sum}}{{{sum(artifact.trit for artifact in artifacts)}}}",
        r"\meta{trit-modulus}{3}",
        rf"\meta{{color-kernel-revision}}{{{provenance.revision}}}",
        rf"\meta{{color-kernel-tree}}{{{provenance.git_tree}}}",
        rf"\meta{{color-kernel-version}}{{{provenance.project_version}}}",
        rf"\meta{{color-kernel-verified-date}}{{{provenance.verified_date}}}",
        r"\meta{color-seed}{stable seed of public artifact alias}",
        r"\meta{color-selection}{first unused color matching the assigned trit}",
        "",
        (
            r"\p{This public page lists every file in the locally verified "
            r"offer surface as a snapshot-scoped anonymous artifact alias and assigns "
            r"each file "
            rf"a deterministic color. It contains {len(artifacts)} files "
            rf"in {unit_count} units: {len(requestable)} requestable artifacts and "
            rf"{len(tooling)} simulated tooling fixtures. "
            r"The exact restricted inventory, filenames, paths, timestamps, hashes, "
            r"accounts, host details, participant-like labels, and row linkages are "
            r"intentionally not embedded.}"
        ),
        (
            r"\p{A listing means that an artifact is locally present and that a governed "
            r"request can be reviewed. It is not a license, consent record, blanket offer, "
            r"or release authorization. Colors identify artifact aliases only; they do "
            r"not identify people, sessions, devices, or physiological states.}"
        ),
        (
            r"\p{Aliases are scoped to the snapshot date and are not cross-snapshot "
            r"identities. Cite the snapshot date with every request.}"
        ),
        r"\subtree{",
        r"\title{Request Data}",
        (
            r"\p{Use the private contact route published on the parent page with subject "
            r"\code{[data request]}. Cite one or more public group, unit, or file "
            r"aliases and the snapshot date from this page. Do not put private "
            r"source values in a public issue or pull request.}"
        ),
        (
            r"\p{State the smallest useful artifact set, intended use, requester and "
            r"collaborators, onward-sharing plan, security controls, retention and "
            r"deletion plan, and the applicable consent, ethics, privacy, rights, and "
            r"release-authority basis. Review may return aggregate metadata, a schema, "
            r"a redacted or derived subset, or nothing.}"
        ),
        r"}",
        r"\subtree{",
        r"\title{Color and GF(3) Method}",
        (
            r"\p{Artifact aliases are ordered deterministically by source class, "
            r"anonymous unit, and artifact kind. Complete triples receive "
            r"\code{-1}, \code{0}, \code{+1}; any trailing aliases receive \code{0}. "
            r"Each alias is assigned a stable seed; the generator selects the first "
            r"unused color whose canonical trit matches the "
            r"assigned trit. The ledger has "
            rf"{trit_counts[-1]} minus, {trit_counts[0]} zero, and "
            rf"{trit_counts[1]} plus assignments, so the trit sum is zero and "
            r"therefore conserved modulo three. Files remain separate colored "
            r"artifacts under their anonymous cohort and unit.}"
        ),
        r"}",
        r"\subtree{",
        r"\title{Complete Available-File Ledger}",
        (
            rf"\p{{The ledger below contains all {len(artifacts)} locally present files, "
            rf"totaling {total_bytes:,} bytes. Real-data artifacts remain request-only. "
            r"Simulated entries are listed for completeness and are tooling fixtures, "
            r"not physiological data.}"
        ),
    ]

    cohort_specs = [
        (
            "OFR-A",
            "EEG cohort (April)",
            (
                "Seventeen requestable files from the companion and support surface "
                "cited by the approved April inventory. Placement is cohort-level "
                "navigation and does not establish a file-to-human linkage."
            ),
            [
                (
                    "OFR-A01",
                    "Group A1 — face files",
                    (
                        "Six requestable face files retained by the final companion "
                        "inventory."
                    ),
                ),
                (
                    "OFR-A02",
                    "Group A2 — April screen files",
                    "Seven requestable April-dated screen files.",
                ),
                (
                    "OFR-A03",
                    "Group A3 — supporting files",
                    (
                        "Four requestable supporting files. Listing does not "
                        "establish redistribution rights."
                    ),
                ),
            ],
        ),
        (
            "OFR-B",
            "Math-proofs cohort (March)",
            (
                "Fourteen requestable screen or audio files grouped at the approved "
                "March cohort level: ten March-dated files and four undated companion "
                "files. Participant-row reconciliation remains separate from this "
                "public offer ledger."
            ),
            [
                (
                    "OFR-B01",
                    "Group B1 — March screen and audio files",
                    "Ten requestable March-dated screen or audio files.",
                ),
                (
                    "OFR-B02",
                    "Group B2 — undated screen files",
                    (
                        "Four requestable undated screen files retained so the available-"
                        "file ledger is complete; no participant-row linkage is asserted."
                    ),
                ),
            ],
        ),
        (
            "OFR-C",
            "Edge Esmeralda",
            (
                "The real-data surface is one direct reading containing forty-six "
                "requestable files in a single reading unit. Eleven simulated tooling "
                "fixtures are listed separately for ledger completeness and are not "
                "part of the reading. Cohort placement does not establish a person-level "
                "linkage."
            ),
            [
                (
                    "OFR-C01",
                    "Group C1 — one direct reading",
                    (
                        "One requestable reading unit containing all forty-six retained "
                        "packet streams, metadata files, decoded arrays, derived event "
                        "streams, and available status rollups. Some downstream files "
                        "are derived-only and cannot reconstruct the packet stream."
                    ),
                ),
                (
                    "OFR-C02",
                    "Group C2 — simulated tooling fixtures",
                    (
                        "Six simulated run units listed so the available-file ledger is "
                        "complete. They must be excluded from physiological or "
                        "participant-level analysis."
                    ),
                ),
            ],
        ),
    ]

    by_class: dict[str, list[Artifact]] = defaultdict(list)
    for artifact in artifacts:
        by_class[artifact.class_code].append(artifact)

    mapped_classes = {
        class_code
        for _cohort_code, _cohort_title, _cohort_description, groups in cohort_specs
        for class_code, _title, _description in groups
    }
    if mapped_classes != set(by_class):
        raise RuntimeError("every public class must map to exactly one cohort")

    for cohort_code, cohort_title, cohort_description, groups in cohort_specs:
        cohort_artifacts = [
            artifact
            for class_code, _title, _description in groups
            for artifact in by_class[class_code]
        ]
        lines.extend(
            [
                r"\subtree{",
                rf"\title{{{cohort_title}}}",
                rf"\meta{{cohort-code}}{{{cohort_code}}}",
                rf"\meta{{unit-count}}{{{len({item.unit_alias for item in cohort_artifacts})}}}",
                rf"\meta{{file-count}}{{{len(cohort_artifacts)}}}",
                rf"\meta{{bytes}}{{{sum(item.size for item in cohort_artifacts)}}}",
                rf"\p{{{cohort_description}}}",
            ]
        )
        for class_code, title, description in groups:
            class_artifacts = by_class[class_code]
            class_units: dict[str, list[Artifact]] = defaultdict(list)
            for artifact in class_artifacts:
                class_units[artifact.unit_alias].append(artifact)
            lines.extend(
                [
                    r"\subtree{",
                    rf"\title{{{title}}}",
                    rf"\meta{{request-code}}{{{class_code}}}",
                    rf"\meta{{unit-count}}{{{len(class_units)}}}",
                    rf"\meta{{file-count}}{{{len(class_artifacts)}}}",
                    rf"\meta{{bytes}}{{{sum(item.size for item in class_artifacts)}}}",
                    rf"\p{{{description}}}",
                ]
            )
            for unit_alias, unit_artifacts in class_units.items():
                lines.extend(
                    [
                        r"\subtree{",
                        rf"\title{{{unit_alias}}}",
                        rf"\meta{{unit-alias}}{{{unit_alias}}}",
                        rf"\meta{{file-count}}{{{len(unit_artifacts)}}}",
                        rf"\meta{{bytes}}{{{sum(item.size for item in unit_artifacts)}}}",
                        r"\ul{",
                    ]
                )
                for artifact in unit_artifacts:
                    lines.append(
                        (
                            rf"\li{{\code{{{artifact.artifact_alias}}} — "
                            rf"\code{{{artifact.kind}}}; format \code{{{artifact.format}}}; "
                            rf"{artifact.size:,} bytes; color \code{{{artifact.color}}}; "
                            rf"trit \code{{{artifact.trit}}}; seed "
                            rf"\code{{0x{artifact.seed:016X}}}; index "
                            rf"\code{{{artifact.color_index}}}.}}"
                        )
                    )
                lines.extend([r"}", r"}"])
            lines.append(r"}")
        lines.append(r"}")

    lines.extend(
        [
            r"}",
            r"\subtree{",
            r"\title{Maintenance and Verification}",
            (
                r"\p{The checked-in generator verifies the canonical raw manifest against "
                r"local bytes, rejects missing or unknown companion types, separates real "
                r"and simulated runs from their recorded flags, requires the expected "
                r"unit and file counts, assigns all colors in one clean color process, "
                r"checks uniqueness and GF(3) conservation, emits only allowlisted public "
                r"fields, and separately writes a mode-0600 private alias-to-source registry "
                r"with integrity hashes. A maintainer must regenerate and review this page whenever "
                r"the private offer surface or color-kernel revision changes.}"
            ),
            r"}",
            "",
        ]
    )
    return lines


def validate_public_document(document: str) -> None:
    forbidden = [
        "FORESTER_RESOURCE_PAYLOAD",
        "Complete Canonical Org Resource",
        "/Users/",
        "/Volumes/",
        "/dev/",
        "file://",
        "localhost",
        "Barton",
        "BrainVision",
        "CGX",
        "Cognionics",
        "Kinesis",
        "Gay.jl",
        "GitHub",
        "BCI Factory",
    ]
    for value in forbidden:
        if re.search(re.escape(value), document, re.IGNORECASE):
            raise RuntimeError(f"public document contains forbidden value: {value}")
    if re.search(r"\b[0-9a-fA-F]{64}\b", document):
        raise RuntimeError("public document contains a private-style SHA-256 value")
    if document.count(r"\li{\code{") != EXPECTED_TOTAL_FILES:
        raise RuntimeError("public document does not list every expected file")


def main() -> None:
    args = parse_args()
    raw = collect_raw_artifacts(args.raw_dir, args.raw_manifest)
    real_tui, simulated_tui = collect_tui_artifacts(args.tui_runs_dir)
    companions = collect_companion_artifacts(args.companion_root)
    artifacts = companions + raw + real_tui + simulated_tui
    assign_aliases_and_trits(artifacts)
    provenance = color_provenance(args.color_project)
    assign_colors(artifacts, args.color_project)
    registry = private_registry_document(artifacts)
    sync_private_registry(args.private_registry, registry, args.check)
    document = "\n".join(forest_lines(artifacts, provenance))
    validate_public_document(document)

    if args.check:
        if not args.target.is_file() or args.target.read_text(encoding="utf-8") != document:
            raise RuntimeError("public Forester inventory is stale")
        action = "verified"
    else:
        args.target.write_text(document, encoding="utf-8")
        action = "generated"

    print(
        f"{action} {args.target}: {len(artifacts)} files, "
        f"{sum(artifact.size for artifact in artifacts)} bytes, "
        f"color kernel {provenance.describe} at verified revision "
        f"{provenance.revision}, tree {provenance.git_tree}"
    )


if __name__ == "__main__":
    main()
