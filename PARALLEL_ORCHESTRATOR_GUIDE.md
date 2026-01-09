# Parallel Orchestrator System Guide

## Overview

The Parallel Orchestrator System runs **multiple Claude Code agents simultaneously** on different tracks within the same roadmap group. This dramatically speeds up development!

## How It Works

### Sequential vs Parallel

**Old System (Sequential):**
- 1 agent works on 1 task at a time
- Group D with 5 tracks = 5 sequential runs
- Total time: 5 × (work + review + test + commit)

**New System (Parallel):**
- Up to 4 agents work simultaneously on different tracks
- Group D with 5 tracks = 2 parallel batches (4 + 1)
- Total time: 2 × (work + review + test + commit)
- **Speed improvement: ~2.5x faster!**

### Intelligence Features

1. **Auto-Detection**: Analyzes ROADMAP.md to find parallel tracks
2. **Smart Claiming**: Each agent claims a different track (no conflicts)
3. **Lock System**: Prevents multiple agents from working on the same track
4. **Synchronized Commits**: All tracks commit individually, push together
5. **Progress Tracking**: Each track has its own detailed log

## Files Created

### Core Scripts

1. **`parallel_orchestrator.sh`** - Main parallel orchestration script
   - Identifies available parallel tracks
   - Launches N agents (up to 4 simultaneous)
   - Each agent works on a different track
   - Coordinates final push to GitHub

2. **`scheduled_parallel_orchestrator.sh`** - Scheduled runner
   - Runs parallel orchestrator every 10 minutes
   - Continues for 6 hours (36 runs)
   - Automatic recovery on failures

3. **`orchestrator.sh`** - Single-track sequential orchestrator
   - Use when you want to run just one task
   - Simpler, more predictable

4. **`setup_github_auth.sh`** - GitHub authentication setup
   - SSH key generation
   - Personal Access Token configuration
   - Required before pushing commits

### Monitoring Scripts

5. **`monitor_parallel.sh`** - Real-time status monitor
   - Shows active orchestrators
   - Displays Claude agent count
   - Shows track logs and progress
   - Lists active track claims
   - Git status and recent commits

6. **`monitor_orchestrator.sh`** - Sequential orchestrator monitor
   - For single-track orchestrator

## Directory Structure

```
multiverse_dive/
├── orchestrator.sh                    # Sequential orchestrator
├── parallel_orchestrator.sh           # Parallel orchestrator
├── scheduled_parallel_orchestrator.sh # Scheduled parallel runner
├── setup_github_auth.sh               # Auth setup
├── monitor_parallel.sh                # Parallel monitor
├── monitor_orchestrator.sh            # Sequential monitor
│
├── .orchestrator_locks/               # Track claim locks
│   ├── group_E_track_1.lock
│   ├── group_E_track_2.lock
│   └── ...
│
├── orchestrator_logs/                 # Individual track logs
│   ├── group_E_track_1.log
│   ├── group_E_track_2.log
│   └── ...
│
├── parallel_orchestrator_runs.log     # Main parallel run log
└── orchestrator_runs.log              # Sequential run log
```

## Usage

### Step 1: Set Up GitHub Authentication (REQUIRED)

```bash
./setup_github_auth.sh
```

Choose SSH keys (recommended) or Personal Access Token.

### Step 2: Test Parallel Orchestrator (Single Run)

```bash
./parallel_orchestrator.sh
```

This will:
- Analyze ROADMAP.md
- Identify Group E (current group)
- Find 5 tracks available
- Launch 4 agents in parallel (Track 1, 2, 3, 4)
- Each agent works independently
- All commit individually
- One final push with all changes

### Step 3: Run Scheduled Parallel Orchestrator

```bash
./scheduled_parallel_orchestrator.sh
```

This runs the parallel orchestrator every 10 minutes for 6 hours.

**Or run in background:**
```bash
nohup ./scheduled_parallel_orchestrator.sh > parallel_console.log 2>&1 &
```

### Step 4: Monitor Progress

```bash
./monitor_parallel.sh
```

Or watch specific track:
```bash
tail -f orchestrator_logs/group_E_track_2.log
```

Or watch main log:
```bash
tail -f parallel_orchestrator_runs.log
```

## Example Output

### Parallel Orchestrator Launch

```
═══════════════════════════════════════════════════════
   Parallel Orchestrator - Multi-Track Builder
═══════════════════════════════════════════════════════

Analyzing ROADMAP.md...
Current Group: E
Available Tracks: 5
Tracks: 1 2 3 4 5

Launching 4 orchestrators in parallel...

[Orchestrator 1/4] Starting on Group E, Track 1
  Log: orchestrator_logs/group_E_track_1.log
  Started (PID: 12345)

[Orchestrator 2/4] Starting on Group E, Track 2
  Log: orchestrator_logs/group_E_track_2.log
  Started (PID: 12346)

[Orchestrator 3/4] Starting on Group E, Track 3
  Log: orchestrator_logs/group_E_track_3.log
  Started (PID: 12347)

[Orchestrator 4/4] Starting on Group E, Track 4
  Log: orchestrator_logs/group_E_track_4.log
  Started (PID: 12348)

═══════════════════════════════════════════════════════
All 4 orchestrators launched!
═══════════════════════════════════════════════════════

Waiting for all orchestrators to complete...

[Track 1] Waiting for PID 12345...
✓ Track 1 completed successfully
[Track 2] Waiting for PID 12346...
✓ Track 2 completed successfully
[Track 3] Waiting for PID 12347...
✓ Track 3 completed successfully
[Track 4] Waiting for PID 12348...
✓ Track 4 completed successfully

═══════════════════════════════════════════════════════
   ✨ All tracks completed successfully! ✨
═══════════════════════════════════════════════════════

Pushing all changes to GitHub...
✓ All changes pushed to GitHub
```

### Monitor Output

```
═══════════════════════════════════════════════════════
   Parallel Orchestrator Status Monitor
═══════════════════════════════════════════════════════

✓ Scheduled orchestrator is RUNNING (PID: 42000)
Active Claude agents: 12
Parallel workers: 4

═══════════════════════════════════════════════════════
Track Logs:
═══════════════════════════════════════════════════════

group_E_track_1.log (45K)
  Last modified: 2026-01-09 12:45:32
  Latest:
    [Track 1] Implementing core/analysis/library/registry.py...
    [Track 1] Writing core/analysis/library/registry.py...
    [Track 1] Registry infrastructure complete!

group_E_track_2.log (38K)
  Last modified: 2026-01-09 12:45:28
  Latest:
    [Track 2] Writing tests for flood algorithms...
    [Track 2] Running test suite...
    [Track 2] Review complete - 24 tests passing

═══════════════════════════════════════════════════════
Active Track Claims:
═══════════════════════════════════════════════════════

✓ group_E_track_1 (PID: 12345)
✓ group_E_track_2 (PID: 12346)
✓ group_E_track_3 (PID: 12347)
✓ group_E_track_4 (PID: 12348)
```

## Parallel Execution Strategy

### Current Group: Group E (Algorithm Foundation)

With 5 tracks available:

**Batch 1** (Tracks 1-4 in parallel):
- Track 1: Registry Infrastructure
- Track 2: Baseline Flood Algorithms
- Track 3: Baseline Wildfire Algorithms
- Track 4: Baseline Storm Algorithms

**Batch 2** (Track 5 sequential):
- Track 5: Algorithm Tests

Each track goes through 3 phases:
1. **Phase 1**: Work Agent (implement)
2. **Phase 2**: Review Agent (test)
3. **Phase 3**: Commit Agent (document + commit)

All tracks commit individually, then push together at the end.

## Advantages

✅ **Speed**: 2-4x faster than sequential
✅ **Isolation**: Each track is independent
✅ **Resilience**: One failure doesn't stop others
✅ **Visibility**: Individual logs per track
✅ **Coordination**: Synchronized final push

## Troubleshooting

### Problem: Authentication Errors

```bash
./setup_github_auth.sh
```

### Problem: Stale Lock Files

```bash
rm -rf .orchestrator_locks/*
```

### Problem: Check Agent Status

```bash
./monitor_parallel.sh
```

### Problem: Kill All Orchestrators

```bash
pkill -f parallel_orchestrator
pkill -f "claude --dangerously"
```

### Problem: Check Specific Track

```bash
tail -f orchestrator_logs/group_E_track_2.log
```

## Performance Expectations

For a typical roadmap group with 4-5 parallel tracks:

- **Sequential**: 5-10 hours per group
- **Parallel (4 agents)**: 2-3 hours per group
- **Scheduled (6 hours)**: Can complete 2-3 groups

## Recommendations

1. **Start with a test run**: Run `./parallel_orchestrator.sh` once to test
2. **Monitor closely**: Use `./monitor_parallel.sh` frequently
3. **Check logs**: Watch individual track logs for issues
4. **Set up auth first**: GitHub authentication is required for pushing
5. **Use background mode**: Run scheduled orchestrator with `nohup`

## Current Status

- **Group A**: ✅ DONE
- **Group B**: ✅ DONE
- **Group C**: ✅ DONE
- **Group D**: ✅ DONE
- **Group E**: 🔄 IN PROGRESS (5 tracks, 4 can run in parallel)
- **Groups F-K**: ⏳ Pending

Next execution will work on Group E tracks in parallel!
