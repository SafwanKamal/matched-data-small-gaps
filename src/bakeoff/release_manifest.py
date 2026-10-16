"""Build the release manifest: SHA-256 of data, frozen items, saved replies, analyses, and of the uncommitted weights and
features, plus the pinned environment. Writes release/SHA256SUMS, release/ENVIRONMENT.txt and release/requirements-lock.txt.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys

from bakeoff.protocol import ROOT

REL = ROOT / "release"
GROUPS = {
    "data": ["data/banking77-oos", "data/clinc-single-domain-oos", "data/1.1", "data/multidogo"],
    "protocols": ["confirmatory_protocol.json", "retrieval_suppression_protocol.json", "jev_rerun_protocol.json", "retrieval_mechanism_protocol.json", "massive_robustness_protocol.json", "gh_protocol.json", "api_protocol.json", "revision_free_protocol.json"],
    "items_and_replies": ["artifacts/confirmatory", "artifacts/retrieval-suppression", "artifacts/jev-rerun", "artifacts/retrieval-mechanism", "artifacts/massive-robustness", "artifacts/gh", "artifacts/posthoc"],
    "specialist_weights_and_features (not in git)": ["artifacts/confirmatory/specialists", "artifacts/confirmatory/features", "artifacts/retrieval-suppression/specialists", "artifacts/retrieval-suppression/features", "artifacts/massive-robustness/specialists", "artifacts/massive-robustness/features", "artifacts/revision/features", "artifacts/robustness"],
}


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    REL.mkdir(exist_ok=True)
    lines, seen = [], set()
    for group, paths in GROUPS.items():
        lines.append(f"# {group}")
        for rel in paths:
            p = ROOT / rel
            if not p.exists():
                lines.append(f"# missing: {rel}")
                continue
            files = [p] if p.is_file() else sorted(q for q in p.rglob("*") if q.is_file())
            for f in files:
                key = f.relative_to(ROOT).as_posix()
                if key in seen or "__pycache__" in key:
                    continue
                seen.add(key)
                lines.append(f"{sha256(f)}  {key}")
        print(f"{group}: done", flush=True)
    (REL / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    if not freeze.strip():
        freeze = subprocess.run(["uv", "pip", "freeze", "--python", sys.executable], capture_output=True, text=True).stdout
    (REL / "requirements-lock.txt").write_text(freeze, encoding="utf-8")
    try:
        import torch

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
        torch_v = torch.__version__
    except Exception:  # noqa: BLE001
        gpu, torch_v = "unknown", "unknown"
    env = [f"python {platform.python_version()}", f"platform {platform.platform()}", f"torch {torch_v}", f"gpu {gpu}", f"files hashed {len(seen)}"]
    (REL / "ENVIRONMENT.txt").write_text("\n".join(env) + "\n", encoding="utf-8")
    print("\n".join(env))


if __name__ == "__main__":
    main()
