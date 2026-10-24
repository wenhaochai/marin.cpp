#!/usr/bin/env python3
"""Replace personal cluster identifiers in tracked text files, in place (the repo is public).

Usage: python3 tools/scrub.py [PATH...]          # rewrite tracked + untracked files (or only PATHs)
       python3 tools/scrub.py --check [PATH...]  # exit 1 and list hits, change nothing
       python3 tools/scrub.py --check --staged   # the pre-commit hook: only staged files
"""
import re, subprocess, sys

_U, _G = "wc" + "9403", "KARTHIK" + "N"  # split so this file does not match itself
REPL = [
    (f"/scratch/gpfs/{_G}/{_U}", "$ROOT"),
    (f"/home/{_U}", "/home/USER"),
    (_U, "USER"),
    (_G, "GROUP"),
    (_G.lower(), "group"),
    ("reself" + "chai@" + "gmail.com", "OWNER_EMAIL"),
]
SECRET = re.compile(r"(ghp_[A-Za-z0-9]{30,}|github_pat_\w{30,}|sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{30,}"
                    r"|WANDB_API_KEY\s*=\s*\w{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY)")

def files():
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--staged" in sys.argv:
        cmd = ["git", "diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR"]
    else:
        cmd = ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", *paths]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return [f for f in out.decode().split("\0") if f]

def main():
    check = "--check" in sys.argv
    bad = 0
    for f in files():
        try:
            s = open(f, encoding="utf-8").read()
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue
        for m in SECRET.finditer(s):
            print(f"SECRET {f}: {m.group(0)[:12]}...")
            bad += 1
        t = s
        for a, b in REPL:
            t = t.replace(a, b)
        if t != s:
            if check:
                print(f"ID {f}")
                bad += 1
            else:
                open(f, "w", encoding="utf-8").write(t)
                print(f"scrubbed {f}")
    sys.exit(1 if bad else 0)

main()
