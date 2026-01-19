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

```bash
source ~/.keychain/*-sh           # Load SSH keychain
git add <files>
git commit -m "Short description"
git push origin main
```

Email: `gpriceless@users.noreply.github.com`
