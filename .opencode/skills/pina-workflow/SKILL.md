---
name: pina-workflow
description: >-
  Orchestrates a complete session from problem definition through trained
  solver. This is the entry point for solving differential equations or
  physics problems with neural networks. Use when the user has a vague
  open-ended request like "I have a differential equation", "I want to solve
  a PDE", "I have a physics problem", "I need to model a system",
  "help me solve an ODE", "I want to use neural networks for my equations",
  "I have some data and equations", or "let's build physics-informed neural
  networks". Also triggers on mentions of specific equations (Navier-Stokes,
  Burgers, Poisson, heat equation, wave equation, etc.) or when the user
  describes a physical system they want to simulate. This skill will guide
  the user through the full workflow step by step, loading the right
  sub-skills at the right time.
license: MIT
compatibility: opencode, codex, claude
metadata:
  audience: users
  workflow: pina-workflow
---

# PINA Workflow — End-to-End Agentic Session

> [!IMPORTANT]
> Read [RULES.md](../RULES.md) before using this skill — it applies to all skills.

Orchestrates a full PINA session. Does **not** duplicate sub-skill content —
each stage below requires actually reading the referenced file(s) in full,
not acting from memory, summary, or a prior pass over this conversation.

## Hard rule for every stage

Before doing anything in a stage:
1. Read every file path listed for that stage, in full, with the `view` tool.
2. If a stage lists multiple sub-skills, read **all** of them — the list is
   mandatory, not illustrative.
3. Do not advance to the next stage until the current stage's checklist
   (below) is satisfied.
4. If you already read a file earlier in this session and the file hasn't
   changed, you may skip re-reading it — but only if you can state which
   stage/turn you read it in. If in doubt, re-read.

## Pipeline

```
Stage 1 — Problem definition    → create-problem
  ├── Domain creation           →   define-domains
  ├── Equation definition       →   define-equations
  └── Condition binding         →   condition-setup
Stage 2 — Model selection       → select-model
Stage 3 — Solver selection      → select-solver
Stage 4 — Trainer configuration → select-trainer
```

## Stage 1 — Problem definition

**Read `../create-problem/SKILL.md` in full now.**

create-problem itself delegates to three sub-skills. Read all three in full,
in this order, following whatever instructions create-problem gives for
sequencing:
- `../define-domains/SKILL.md`
- `../define-equations/SKILL.md`
- `../condition-setup/SKILL.md`

**Exception:** if the user already has a working `problem` object, skip
directly to Stage 2 — do not read Stage 1 files.

**Stage 1 checklist (must all be true before moving on):**
- [ ] create-problem read in full
- [ ] define-domains read in full, domains defined
- [ ] define-equations read in full, equations defined
- [ ] condition-setup read in full, conditions bound
- [ ] a `problem` object exists

## Stage 2 — Model selection

**Read `../select-model/SKILL.md` in full now.**

Input/output dimensions are known from the Stage 1 `problem` object — use
them, don't re-derive or guess.

**Stage 2 checklist:**
- [ ] select-model read in full
- [ ] model instantiated with correct input/output dims

## Stage 3 — Solver selection

**Read `../select-solver/SKILL.md` in full now.**

Check the condition types from the `problem` object (set in Stage 1) — they
drive which solver is valid. Do not pick a solver without checking this.

**Stage 3 checklist:**
- [ ] select-solver read in full
- [ ] condition types checked against solver requirements
- [ ] solver instantiated

## Stage 4 — Trainer configuration

**Read `../select-trainer/SKILL.md` in full now.**

Verify all domains are discretised before configuring the trainer — if not,
go back to define-domains, don't silently proceed.

**Stage 4 checklist:**
- [ ] select-trainer read in full
- [ ] all domains confirmed discretised
- [ ] trainer configured

## Stage 5 — Train

The user runs `trainer.train()`. Offer to inspect results afterwards
(forward pass, loss curves, discovered unknown parameters for inverse
problems).