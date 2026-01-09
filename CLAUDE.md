# Claude Context

Geospatial event intelligence platform that converts (area, time window, event type) into decision products.

## Core Concept

Situation-agnostic specifications enable the same agent-orchestrated pipelines to handle any hazard type (flood, wildfire, storm) without bespoke logic.

## Architecture

- **OpenSpec**: JSON Schema + YAML for event, intent, data source, pipeline, and provenance specifications
- **Data Broker**: Multi-source discovery with constraint evaluation, ranking, and open-source preference
- **Analysis Layer**: Algorithm library with dynamic pipeline assembly and hybrid rule/ML selection
- **Quality Control**: Automated sanity checks, cross-validation, consensus generation

## Key Files

- `OPENSPEC.md` - Complete system design specification
- `ROADMAP.md` - Prioritized implementation roadmap with parallel work groups
- `FIXES.md` - **Required bug fixes** with exact code changes (check before implementing new features)

## Before Starting New Work

1. **Check FIXES.md** for any P0 (critical) bugs that must be fixed first
2. **Check ROADMAP.md** for current progress and what can be parallelized
3. Run tests: `PYTHONPATH=. .venv/bin/pytest tests/ -v`

## Git Workflow

**Push regularly after completing work.** The repo is configured for GitHub.

```bash
# Stage and commit
git add <files>
git commit -m "$(cat <<'EOF'
Short description of changes

- Detail 1
- Detail 2

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"

# Push to GitHub
git push origin main
```

**Note:** Email is configured as `gpriceless@users.noreply.github.com` for GitHub privacy compliance.

## Current Status

- Groups A-D: Complete (schemas, validator, intent, data discovery)
- Group E: ~60% complete (flood algorithms done, wildfire/storm pending)
- Groups F-K: Not started
- **16 bugs documented** in FIXES.md (6 critical, 5 medium, 5 low priority)
