# FirstLight lessons

Read at session start. Add an entry after every resolved bug. At most 40 entries. A
lesson seen twice becomes a CLAUDE.md line, a lint or a hook and leaves this file; what
left is kept in `history/lessons-evicted.md`.

## Patterns

None open. Promoted so far: inputs chosen by sort order or nearest date, checks that
cannot fail, removing an export, and stand-ins for a failed lookup.

## Individual

### zarr 3 removed DirectoryStore
- Use `zarr.storage.LocalStore`, `create_array` and `zarr_format=2`, because numcodecs
  compressors and xarray's `_ARRAY_DIMENSIONS` attribute need format 2. Through xarray
  the encoding key is `compressors`, a list.

### A fixed chooser had a twin
- Wildfire scene choice was fixed to rank by clear share; storm kept its own copy of
  the old nearest-date pick for a day. Before closing a fix to a picker, grep for its
  siblings (`_find_*_scenes`) and route them through the fixed one, then delete them.
