#!/usr/bin/env python3
"""
Apply your YAML patch to the core theme dictionary JSON.
"""
import json
import argparse
import pathlib

def merge(core_dict, patch_dict):
    """
    Given two dicts mapping theme→config,
    overlay the 'concepts_add' and 'negative_keywords_add'
    onto the core, returning the updated core.
    """
    for theme, patch_cfg in patch_dict.items():
        core_cfg = core_dict.setdefault(theme, {})
        # Merge keywords
        for key in ("concepts_add", "negative_keywords_add"):
            additions = patch_cfg.get(key, [])
            if additions:
                core_list = set(core_cfg.get(key.replace("_add",""), []))
                core_list.update(additions)
                core_cfg[key.replace("_add","")] = sorted(core_list)
    return core_dict

def main(core_path, patch_path, out_path):
    core = json.loads(pathlib.Path(core_path).read_text(encoding="utf-8"))
    patch = json.loads(pathlib.Path(patch_path).read_text(encoding="utf-8"))
    merged = merge(core, patch)
    pathlib.Path(out_path).write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Patched {len(patch)} themes → {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge theme patch")
    parser.add_argument("--core",  "-c", dest="core",  required=True)
    parser.add_argument("--patch", "-p", dest="patch", required=True)
    parser.add_argument("--out",   "-o", dest="out",   required=True)
    args = parser.parse_args()
    main(args.core, args.patch, args.out)
