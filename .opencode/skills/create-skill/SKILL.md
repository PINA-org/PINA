---
name: create-skill
description: >-
  Create, modify, and improve PINA skills. Use when the user wants to create a
  new skill for PINA development guidance, edit an existing PINA skill, or add
  cross-tool compatibility (opencode, Claude Code, Codex CLI) to a skill.
  Covers skill directory structure, SKILL.md front matter conventions, symlinks
  for Claude Code compatibility, and updating AGENTS.md for Codex CLI.
license: MIT
compatibility: opencode, codex, claude
metadata:
  audience: users
  workflow: skill-creation
---

# Create a PINA Skill

> [!IMPORTANT]
> Read [RULES.md](../RULES.md) before using this skill — it applies to all skills.

Use this skill to create new PINA development skills or modify existing ones.
Each PINA skill lives at `.opencode/skills/<name>/SKILL.md` and must be made
available to all three tools (opencode, Claude Code, Codex CLI).

## Overview

A PINA skill is a markdown file with YAML front matter that provides guided
instructions for a specific PINA development task. Skills are auto-discovered
by opencode, symlinked for Claude Code, and discoverable via the root
`AGENTS.md` for Codex CLI.

### What makes a good PINA skill

- **Narrow scope** — One focused task (e.g. defining domains, not "all of PINA")
- **Concrete templates** — Copy-paste-ready code snippets using PINA's API
- **Checklists** — Verifiable steps so the user knows when they're done
- **Progressive disclosure** — Start with the common case, then branch to edge cases
- **Why, not just what** — Explain why PINA conventions exist (e.g. why operators trigger a backward pass per call)

### Anatomy of a skill

```
.opencode/skills/<name>/
├── SKILL.md (required)
│   ├── YAML front matter (name, description, license, compatibility, metadata)
│   └── Markdown instructions
└── (optional resources)
```

## Step 1 — Capture intent

Understand what the user wants the skill to do:

1. **What PINA task should this skill guide?** (e.g. "setting up boundary conditions", "defining custom PDEs", "discretising domains")
2. **Who is the audience?** (new PINA users, experienced users, or both)
3. **What existing skill is closest?** — Check `.opencode/skills/` for related skills. If this skill extends an existing workflow, reference it.
4. **What PINA APIs does it need to cover?** — List the classes, functions, and import paths.

## Step 2 — Study existing skills as templates

Read the existing PINA skills for style and structure. Key patterns:

| File | Purpose | Reference pattern |
|------|---------|-------------------|
| `create-problem/SKILL.md` | Entry point, decision tree, templates | Multiple templates per audience, step-by-step flow, checklist |
| `define-equations/SKILL.md` | Equation definition | Zoo reference table, custom signatures, edge cases |
| `define-domains/SKILL.md` | Domain creation | Domain types table, common patterns, discretisation |
| `condition-setup/SKILL.md` | Condition binding | Condition types table, data type options |

### Front matter template

Use the same front matter format as existing skills:

```yaml
---
name: <kebab-case-name>
description: >-
  One or two sentences describing when this skill triggers. Include specific
  user phrases and contexts that should activate it. Be slightly "pushy" —
  err on the side of including near-miss use cases so the skill triggers when
  it should.
license: MIT
compatibility: opencode, codex, claude
metadata:
  audience: users
  workflow: <workflow-name>
---
```

### Writing guidelines

- **Imperative voice** — "Define the domains", "Create the condition dict"
- **Explain the why** — LLMs work better when they understand the reasoning behind conventions
- **Use tables** for API references and option lists (consistent with existing skills)
- **Provide templates** — At least one complete code template the user can adapt
- **Checklists** at the end — Verifiable completion criteria
- **Front matter `description`** is the primary trigger — include when-to-use context that matches real user phrasing

## Step 3 — Write the SKILL.md

Organise content with progressive disclosure:

```markdown
# [Title]

> [!IMPORTANT]
> Read [RULES.md](../RULES.md) before using this skill — it applies to all skills.
> (If sub-skill: This is a sub-skill of **parent-skill**. Load the entry-point skill first.)

Use this skill to [concise purpose statement].

## Step 1 — [First step]
...
## Step N — [Last step]

## Templates
...
## Checklist
- [ ] ...
```

### Cross-tool compatibility

Every new skill **must** be made available to all three tools. Include this
checklist item and guide the user through the steps:

1. **opencode** — The skill at `.opencode/skills/<name>/SKILL.md` is auto-discovered. No extra step needed.
2. **Claude Code** — Create a symlink:
   ```bash
   mkdir -p .claude/skills/<name>
   ln -sf ../../../.opencode/skills/<name>/SKILL.md .claude/skills/<name>/SKILL.md
   ```
3. **Codex CLI** — Create a symlink (same pattern):
   ```bash
   mkdir -p .agents/skills/<name>
   ln -sf ../../../.opencode/skills/<name>/SKILL.md .agents/skills/<name>/SKILL.md
   ```
   The relative path `../../../.opencode/skills/<name>/SKILL.md` works from both symlink directories (3 `..` to root, then into `.opencode/`).

## Step 4 — Verify cross-tool setup

After writing the skill, verify both symlinks resolve:

```bash
test -f .claude/skills/<name>/SKILL.md && echo "claude: OK"
test -f .agents/skills/<name>/SKILL.md && echo "codex: OK"
```

## Step 5 — Review and iterate

Present the skill to the user for review. Check:

- Does the front matter `description` clearly convey when to use this skill?
- Are the templates syntactically correct PINA code?
- Is the cross-tool compatibility checklist complete?
- Does the skill reference [RULES.md](../RULES.md) and link to related skills?

## Templates

### Template 1: New PINA skill (minimal)

```yaml
---
name: my-new-skill
description: >-
  Guides users through [task]. Use when the user asks about [topic] or needs
  help with [specific problem].
license: MIT
compatibility: opencode, codex, claude
metadata:
  audience: users
  workflow: my-workflow
---
```

### Template 2: Symlinks for all three tools

After creating `.opencode/skills/my-new-skill/SKILL.md`:
```bash
mkdir -p .claude/skills/my-new-skill
ln -sf ../../../.opencode/skills/my-new-skill/SKILL.md .claude/skills/my-new-skill/SKILL.md
mkdir -p .agents/skills/my-new-skill
ln -sf ../../../.opencode/skills/my-new-skill/SKILL.md .agents/skills/my-new-skill/SKILL.md
test -f .claude/skills/my-new-skill/SKILL.md && echo "claude: OK"
test -f .agents/skills/my-new-skill/SKILL.md && echo "codex: OK"
```

## Checklist

- [ ] `name` is kebab-case, unique across `.opencode/skills/`
- [ ] `description` front matter includes trigger context (when to use)
- [ ] `compatibility: opencode, codex, claude` is set
- [ ] `metadata.audience` and `metadata.workflow` are set
- [ ] Skill references [RULES.md](../RULES.md) at the top
- [ ] Sub-skills reference their parent entry-point skill
- [ ] At least one complete code template is provided
- [ ] Cross-tool setup is complete:
  - [ ] `.claude/skills/<name>/SKILL.md` symlink exists and resolves
  - [ ] `.agents/skills/<name>/SKILL.md` symlink exists and resolves
