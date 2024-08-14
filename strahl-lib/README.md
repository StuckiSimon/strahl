# strahl-lib

Main library package provided as [NPM package](https://www.npmjs.com/package/strahl).

For documentation on `strahl`, please refer to the [Github repository](https://github.com/StuckiSimon/strahl).
This file provides information about how to work on the `strahl` library itself, and not how to use it.

## Development

`npm run dev`

In order to expose in local network, run:

`npm run dev -- --host`

### Testing

`npm run test`

### Access

- [https://localhost:5173/](https://localhost:5173/) - Demo Scene
- [https://localhost:5173/benchmark.html](https://localhost:5173/benchmark.html) - Benchmark Setup

### Publish

1. `git stash --include-untracked`
1. `npm run build`
1. `npm version prerelease --preid=alpha`
1. `npm publish --dry-run`
1. `npm publish`
