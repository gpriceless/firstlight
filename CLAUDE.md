<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

# Claude Context - FirstLight

Geospatial event intelligence platform that converts (area, time window, event type) into decision products.

## Quick Reference

| Item | Location |
|------|----------|
| Bug fixes | `FIXES.md` |
| Roadmap & tasks | `ROADMAP.md` |
| Agent memory | `.claude/agents/PROJECT_MEMORY.md` |
| Active specs | `OPENSPEC.md` |
| Completed specs | `docs/OPENSPEC_ARCHIVE.md` |

## Current Status

- **Core Platform:** Complete (170K+ lines, 518+ tests)
- **P0 Bugs:** All fixed - platform is production-ready

### Work Streams Available

1. **Image Validation** - Band validation before processing (COMPLETE)
2. **Distributed Processing** - Dask parallelization for large rasters (COMPLETE)

## Before Starting Work

1. Read `.claude/agents/PROJECT_MEMORY.md` for context
2. Check `FIXES.md` for any new bugs
3. Run tests: `./run_tests.py` or `./run_tests.py <category>`

## Test Commands

```bash
./run_tests.py                    # All tests
./run_tests.py flood              # Flood tests
./run_tests.py wildfire           # Wildfire tests
./run_tests.py schemas            # Schema tests
./run_tests.py --algorithm sar    # Specific algorithm
./run_tests.py --list             # Show categories
```

## CLI Commands

```bash
flight --help                     # Show all commands
flight info                       # System info
flight discover --area area.geojson --event flood  # Find data
flight run --area area.geojson --event flood --profile laptop  # Run analysis
```

## Git Workflow

Follow standard git workflow. See `GUIDELINES.md` for details.
