#!/usr/bin/env python3
"""
Apply your YAML patch to the core theme dictionary JSON.
"""

import json
import argparse
import pathlib

def merge(core_dict, patch_dict):
    """
    Overlay patch_dict entries into core_dict.
    Supports 'concepts_add' and 'negative_keywords_add'.
    """
    for theme, patch_cfg in patch_dict.items():
        core_cfg = core_dict.setdefault(theme, {})
        # Handle new keywords
        for key in ("concepts_add", "negative_keywords_add"):
            adds = patch_cfg.get(key, [])
            if adds:
                base_key = key.replace("_add", "")
                existing = set(core_cfg.get(base_key, []))
                existing.update(adds)
                core_cfg[base_key] = sorted(existing)
    return core_dict

def main(core_path, patch_path, out_path):
    core = json.loads(pathlib.Path(core_path).read_text(encoding="utf-8"))
    patch = json.loads(pathlib.Path(patch_path).read_text(encoding="utf-8"))
    merged = merge(core, patch)
    pathlib.Path(out_path).write_text(
        json.dumps(merged, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"✅ Patched {len(patch)} themes → {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge theme patch YAML")
    parser.add_argument(
        "--core", "-c", dest="core", required=True,
        help="path to theme_dict_core.json"
    )
    parser.add_argument(
        "--patch", "-p", dest="patch", required=True,
        help="path to theme_dict_patch.yaml"
    )
    parser.add_argument(
        "--out", "-o", dest="out", required=True,
        help="where to write updated core JSON"
    )
    args = parser.parse_args()
    main(args.core, args.patch, args.out)
