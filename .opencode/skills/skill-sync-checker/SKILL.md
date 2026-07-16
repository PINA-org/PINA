---
name: skill-sync-checker
description: Audits SKILL.md files in this repo's skills directory for references to functions, classes, or modules (mentioned by name in prose, e.g. `parse_invoice`) that no longer exist in the codebase because they were renamed, moved, or deleted. Proposes a targeted fix to the one skill being checked, never a blanket rewrite. ONLY use this skill when the user's message begins with the exact phrase "I am a pina developer and I need to check skills are updated with code". This is a deliberate, narrow, internal-only trigger — do not activate this skill for generic requests like "check if my docs are outdated," "review my skills," "are these skills still accurate," or any similar phrasing that lacks that exact opening phrase, even if the underlying intent sounds similar.
---

# Skill Sync Checker

Skills in this repo describe parts of the codebase by name (function names, class
names, module names) rather than by file path. Code drifts — things get renamed,
moved, deleted — and the prose in a SKILL.md doesn't automatically follow along.
This skill catches that drift and helps fix it, one skill at a time, with the
user confirming before anything is written.

## Step 0: Confirm this is the right moment

This skill should only ever be invoked when the user's message opens with the
exact phrase "I am a pina developer and I need to check skills are updated with
code". If you've somehow ended up here without that phrase having been used,
stop and ask the user to confirm that's really what they want — this skill's
edits touch other skills' documentation and shouldn't run casually.

## Step 1: Figure out the scope

Read the rest of the user's message after the trigger phrase for a specific
skill name. Two cases:

- **A skill is named** → skip straight to Step 3 for that skill.
- **No skill is named** → run the overview scan first (Step 2) and ask the user
  which flagged skill they want to dig into. Don't deep-check every flagged
  skill in one pass — this skill fixes one skill at a time, by design, so a
  single bad suggestion can't cascade across the repo.

## Step 2: Overview scan (only when no skill was named)

Run the script in overview mode to see drift counts across every skill:

```bash
python3 utils/check_skill_sync.py --skills-dir <path-to-skills-dir> --code-dir <path-to-code-dir> --overview
```

You'll need to figure out `<path-to-skills-dir>` and `<path-to-code-dir>` from
the repo layout — look for a `skills/` folder and the actual source folder
(commonly `src/`, but check first). Present the table to the user in plain
language and ask which skill they'd like to check in depth.

## Step 3: Deep check the named skill

```bash
python3 utils/check_skill_sync.py --skills-dir <path-to-skills-dir> --code-dir <path-to-code-dir> --skill <skill-name> --json-out /tmp/sync_report.json
```

This prints a human-readable report and also writes JSON to `/tmp/sync_report.json`
so you can reason over it precisely. The report buckets every code-like
reference found in that skill's SKILL.md into:

- **defined** — confirmed to exist in the codebase, no action needed.
- **used_only** — the token shows up somewhere in the code but not as a
  definition the script could recognize (e.g. it's imported, not declared, or
  it's a language/pattern the script's heuristics don't cover well). Lower
  confidence — worth a quick manual look, not necessarily worth flagging to
  the user.
- **not_found** — didn't turn up anywhere in `--code-dir`. This is the
  interesting bucket.

Read the "Notes / heuristics" section at the top of the script before trusting
`not_found` results — it only flags things that *look* like code identifiers
(underscored, mixed-case, dotted, or written as a call), so plain English
words in backticks are already filtered out. But `not_found` can still mean
"exists but outside the folder you pointed --code-dir at" rather than
"actually gone" — so verify, don't just trust the label.

## Step 4: Investigate each `not_found` item

For every flagged token, before proposing anything:

1. Check the script's `suggestions` field (fuzzy matches against real symbol
   names) — often this is exactly the rename.
2. If there's no confident suggestion, search the codebase yourself (grep for
   a substring, check git history/blame if available) to figure out what
   actually happened to it: renamed, moved to another module, or genuinely
   removed.
3. If you can't find a confident answer, say so plainly rather than guessing
   — it's better to tell the user "I can't tell what happened to `X`" than to
   propose a fix built on a guess.

## Step 5: Propose the edit, then wait for confirmation

For each `not_found` item you've resolved, show the user the specific before/after
text change for that one skill's SKILL.md — a small diff-style snippet, not a
full rewrite of the file. Something like:

```
`old_helper_name()` → `helper_name_v2()`   (renamed in src/utils.py, moved during the 2026-04 refactor)
```

Wait for explicit confirmation before writing anything. If the user only
confirms some of the proposed changes, apply only those.

**Never touch a skill other than the one named in this session**, even if the
overview scan in Step 2 showed drift elsewhere. If other skills are also
flagged, just remind the user they can run this again naming that skill.


## Limitations worth knowing

- The script has no real parser — it's regex-based heuristics across common
  languages. It will occasionally miss genuine drift (false negative) or flag
  something that's actually fine, like a third-party/library symbol that
  isn't defined in this repo or new built code as example on top of pina
  (false positive labeled `used_only` or `not_found`).
- Single-word, all-lowercase identifiers with no underscore (e.g. a function
  literally named `parse`) are deliberately not flagged from prose, to avoid
  drowning real findings in false positives from ordinary English words in
  backticks. If a skill leans on names like that, mention it explicitly in
  the skill text with an underscore or in a fenced code block so the script
  can see it.