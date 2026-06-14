# NNNN — Short title (verb phrase or "X over Y")

## Decision

One sentence: what we picked.

## Status

One of:

- `accepted` — current behaviour, evidence-backed.
- `provisional` — current behaviour, but re-evaluate after experiment X.
- `superseded by NNNN-other-decision.md` — see linked successor.
- `deferred` — no decision yet; this doc captures the open question and
  the experiment that would resolve.

## Context

What problem we faced. What constraints applied (hardware, brief rules,
upstream choices, time-box). 2-5 sentences. Cite `LESSONS.md §N`,
`CHANGELOG.md vX.Y.Z`, `PROJECT_BRIEF.md §N`, or `path/to/file.py:NN`
where the rationale already lives.

## Decision

One sentence repeat — same wording as the top of the file.

## Alternatives considered

- **Alternative A** — one-line trade-off (why it lost).
- **Alternative B** — one-line trade-off.
- **Alternative C** — one-line trade-off (optional; cap at 4).

## Rationale

Why we picked what we did. 3-8 sentences. Cite measurements, retro
findings, or constraints rather than asserting. If no documented
rationale exists at decision time, set status to `provisional` and
write the experiment in the *Re-litigation triggers* section.

## Trade-offs accepted

What we gave up by choosing this path. Be honest. If there's a known
gap (latency penalty, dependency surface, sudo prereq, dead-code
field), state it.

## Re-litigation triggers

What evidence would make us revisit this decision. Examples:

- Hardware change (new GPU class, new CPU, different OS).
- Upstream/library version bumps (e.g. ORT 1.25 fixes the int64 issue).
- A measured listener-test review that contradicts the current default.
- A new alternative entering the ecosystem (e.g. a successor to RVC).

If revisiting is unlikely on any realistic timescale, write
"None foreseen on the current stack" and explain why the constraint is
permanent.

---

## How to use this template

1. Copy `0000-template.md` to `NNNN-short-title.md` where `NNNN` is the
   next free four-digit number (zero-padded, monotonic).
2. Fill every section. Don't pad — 50-200 lines is the target.
3. Don't fabricate citations. If you can't cite, mark status
   `provisional` and write the experiment that would resolve.
4. Update `README.md` (this directory's index) to add the new entry.
5. Cross-link from any code comment that references this decision.
