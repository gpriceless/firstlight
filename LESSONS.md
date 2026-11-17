# FirstLight lessons

Read at session start. Add an entry after every resolved bug. At most 40 entries. A
lesson seen twice becomes a CLAUDE.md line, a lint or a hook and leaves this file; what
left is kept in `history/lessons-evicted.md`.

## Patterns

None open. Promoted so far: inputs chosen by sort order or nearest date, checks that
cannot fail, removing an export, stand-ins for a failed lookup, and a fix made to one
hazard's path but not its twin.

## Individual

### zarr 3 removed DirectoryStore
- Use `zarr.storage.LocalStore`, `create_array` and `zarr_format=2`, because numcodecs
  compressors and xarray's `_ARRAY_DIMENSIONS` attribute need format 2. Through xarray
  the encoding key is `compressors`, a list.
