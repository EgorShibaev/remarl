"""Public artifact downloads with pinned revisions and checksum verification."""

import json
from pathlib import Path
import shutil
import ssl
import tempfile
import urllib.request
import zipfile
import truststore
from .io import sha256
from .runtime import BACKBONE_ID, BACKBONE_REVISION, load_policy

RESOURCES = Path(__file__).parent / "resources"


def checkpoint_archives(manifest):
    if "archives" in manifest:
        return manifest["archives"]
    return [dict(url=manifest["archive_url"], sha256=manifest["archive_sha256"],
                 files=[e["file"] for e in manifest["checkpoints"]])]


def fetch(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    with (
        urllib.request.urlopen(url, context=context, timeout=60) as response,
        temporary.open("wb") as out,
    ):
        shutil.copyfileobj(response, out)
    temporary.replace(destination)


def verify(directory, filenames=None):
    manifest = json.loads((RESOURCES / "checkpoints.json").read_text())
    verified = []
    for entry in manifest["checkpoints"]:
        if filenames is not None and entry["file"] not in filenames:
            continue
        path = Path(directory) / entry["file"]
        if not path.exists() or sha256(path) != entry["sha256"]:
            raise ValueError(f"Missing or incorrect checkpoint: {path}")
        _, saved = load_policy(path)
        if saved["policy_sha256"] != entry["policy_sha256"]:
            raise ValueError(f"Incorrect policy: {path}")
        verified.append(entry["file"])
    return dict(verified=verified)


def install_checkpoints(archive, output):
    manifest = json.loads((RESOURCES / "checkpoints.json").read_text())
    checksum = sha256(archive)
    archives = checkpoint_archives(manifest)
    matches = [a for a in archives if a["sha256"] == checksum]
    if len(matches) != 1:
        raise ValueError("Checkpoint archive checksum mismatch")
    filenames = set(matches[0]["files"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Extract only named policy files, never arbitrary paths or optimizer helpers.
    with zipfile.ZipFile(archive) as bundle:
        for entry in manifest["checkpoints"]:
            if entry["file"] not in filenames:
                continue
            matches = [n for n in bundle.namelist() if Path(n).name == entry["file"]]
            if len(matches) != 1:
                raise ValueError(f"Missing/ambiguous checkpoint {entry['file']}")
            destination = output / entry["file"]
            if destination.exists() and sha256(destination) != entry["sha256"]:
                raise FileExistsError(f"Refusing to overwrite different weights: {destination}")
            payload = bundle.read(matches[0])
            import hashlib

            if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                raise ValueError(f"Checkpoint checksum mismatch: {entry['file']}")
            destination.write_bytes(payload)
    return verify(output, filenames)


def download(asset, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if asset in ("backbone", "dream-backbone"):
        from huggingface_hub import snapshot_download

        from .dream import DREAM_ID, DREAM_REVISION

        model_id, revision = (
            (DREAM_ID, DREAM_REVISION)
            if asset == "dream-backbone"
            else (BACKBONE_ID, BACKBONE_REVISION)
        )
        snapshot_download(model_id, revision=revision, local_dir=output)
        (output / "remarl-model.json").write_text(
            json.dumps(dict(model_id=model_id, revision=revision), indent=2) + "\n"
        )
    elif asset == "checkpoints":
        import gdown

        manifest = json.loads((RESOURCES / "checkpoints.json").read_text())
        archives = checkpoint_archives(manifest)
        for item in archives:
            try:
                verify(output, item["files"])
                continue
            except ValueError:
                pass
            with tempfile.TemporaryDirectory() as directory:
                archive = Path(directory) / "checkpoints.zip"
                # Public download without browser cookies or a Drive login.
                file_id = item["url"].split("/d/", 1)[1].split("/", 1)[0]
                gdown.download(id=file_id, output=str(archive), use_cookies=False)
                install_checkpoints(archive, output)
        print(json.dumps(verify(output), indent=2))
    elif asset in ("math-data", "code-data"):
        for entry in json.loads((RESOURCES / "datasets.json").read_text()):
            if entry["domain"] != asset.removesuffix("-data"):
                continue
            destination = output / (entry["name"] + Path(entry["file"]).suffix)
            url = f"https://huggingface.co/datasets/{entry['repo']}/resolve/{entry['revision']}/{entry['file']}"
            if not destination.exists():
                fetch(url, destination)
            if sha256(destination) != entry["sha256"]:
                raise ValueError(f"Dataset checksum mismatch: {destination}")
            print(destination)
    elif asset == "baseline-sources":
        from .baselines.sources import download_sources

        download_sources(output)
    else:
        raise ValueError(asset)
