"""Build <team_name>_submission.zip with the required structure:

    output/matching_results.tsv
    output/candidate_pairs.tsv
    code/business_entity_resolution/src/*.py
    code/business_entity_resolution/README.md
    code/business_entity_resolution/requirements.txt
    Documentation_template.md
"""
import argparse
import os
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))  # repo root (src -> ber -> code -> root)


def add(zf, abspath, arcname):
    if os.path.exists(abspath):
        zf.write(abspath, arcname)
        return True
    print("  MISSING:", arcname)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default="team")
    ap.add_argument("--root", default=ROOT)
    a = ap.parse_args()
    root = a.root
    zip_path = os.path.join(root, f"{a.team}_submission.zip")
    ber = os.path.join("code", "business_entity_resolution")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        add(zf, os.path.join(root, "output", "matching_results.tsv"),
            "output/matching_results.tsv")
        add(zf, os.path.join(root, "output", "candidate_pairs.tsv"),
            "output/candidate_pairs.tsv")
        src_dir = os.path.join(root, ber, "src")
        for fn in sorted(os.listdir(src_dir)):
            if fn.endswith(".py"):
                add(zf, os.path.join(src_dir, fn), f"{ber}/src/{fn}".replace("\\", "/"))
        add(zf, os.path.join(root, ber, "README.md"), f"{ber}/README.md".replace("\\", "/"))
        add(zf, os.path.join(root, ber, "requirements.txt"),
            f"{ber}/requirements.txt".replace("\\", "/"))
        add(zf, os.path.join(root, "Documentation_template.md"), "Documentation_template.md")
    size = os.path.getsize(zip_path)
    print(f"wrote {zip_path} ({size/1024:.1f} KiB)")
    with zipfile.ZipFile(zip_path) as zf:
        for n in zf.namelist():
            print("  ", n)


if __name__ == "__main__":
    main()
