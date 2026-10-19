"""Package the source into a directly installable, reproducible Blender add-on ZIP."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

EXCLUDED = {'.git', '.github', '.gitignore', '.python_dependencies', '__pycache__'}


def selected_files(source):
    return sorted(
        path for path in source.rglob('*')
        if path.is_file()
        and not path.is_symlink()
        and not EXCLUDED.intersection(path.relative_to(source).parts)
        and path.suffix not in {'.pyc', '.pyo'}
    )


def build(source, output):
    source, output = source.resolve(), output.resolve()
    if not (source / '__init__.py').is_file() or not (source / 'compatibility_installer.py').is_file():
        raise ValueError('The source must be the complete dream_textures add-on folder.')
    if output.is_relative_to(source):
        raise ValueError('Write the release ZIP outside the source folder.')
    manifest_path = source / 'compatibility/BUILD_MANIFEST.json'
    manifest = {
        'build': '0.4.2-compat.2',
        'built_on': '2026-10-19',
        'upstream_commit': 'c2622a8a9f1ae6b790cfe1d2571f814b126811b4',
        'target': {'blender': '5.2.2', 'python': '3.13', 'platform': 'Windows x64', 'gpu': 'RTX 5060'},
        'files': {
            path.relative_to(source).as_posix(): {
                'bytes': path.stat().st_size,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in selected_files(source) if path != manifest_path
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in selected_files(source):
            info = zipfile.ZipInfo('dream_textures/' + path.relative_to(source).as_posix(), (2026, 10, 19, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError('ZIP checksum failure: ' + bad)
        entries = len(archive.infolist())
    result = {
        'file': str(output), 'bytes': output.stat().st_size, 'entries': entries,
        'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.output)
