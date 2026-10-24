#!/usr/bin/env python3
"""Find (and, with --write, replace) personal cluster identifiers and secrets in text files (the repo is public).

Usage: python3 tools/scrub.py [PATH...]              # check only (the default): exit 1 and list hits
       python3 tools/scrub.py --check --staged       # the pre-commit hook: only staged files
       python3 tools/scrub.py --write PATH [PATH...] # rewrite ONLY the named paths, in place
-h/--help prints this text; any other option is an error. --write needs explicit paths and never touches bench/
or cache/: their scripts need real paths to run (a blanket --write once broke 46 bench scripts, 2026-10-24).
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

FLAGS = {"--check", "--staged", "--write"}


def usage_and_exit(code):
    print(__doc__.strip())
    sys.exit(code)


def files():
    paths = [a for a in sys.argv[1:] if not a.startswith("-")]
    if "--staged" in sys.argv:
        cmd = ["git", "diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR"]
    else:
        cmd = ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", *paths]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return [f for f in out.decode().split("\0") if f]

def main():
    opts = [a for a in sys.argv[1:] if a.startswith("-")]
    if any(o in ("-h", "--help") for o in opts):
        usage_and_exit(0)
    unknown = [o for o in opts if o not in FLAGS]
    if unknown or ("--staged" in opts and "--check" not in opts):
        print("scrub.py: " + ("unknown option " + " ".join(unknown) if unknown else "--staged needs --check"),
              file=sys.stderr)
        usage_and_exit(2)
    write = "--write" in opts
    paths = [a for a in sys.argv[1:] if not a.startswith("-")]
    if write and ("--check" in opts or not paths or any(p.rstrip("/").split("/")[0] in ("bench", "cache") for p in paths)):
        print("scrub.py: --write needs explicit paths outside bench/ and cache/, and excludes --check", file=sys.stderr)
        usage_and_exit(2)
    check = not write
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
