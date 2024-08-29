# strahl-lib

Main library package provided as [NPM package](https://www.npmjs.com/package/strahl).

For documentation on `strahl`, please refer to the [Github repository](https://github.com/StuckiSimon/strahl).
This file provides information about how to work on the `strahl` library itself, and not how to use it.

## Development

`npm run dev`

In order to expose in local network, run:

`npm run dev -- --host`

### Testing

The project uses `vitest` for unit testing.

`npm run test`

### Linting

The project uses `eslint` for static code analysis.

`npm run lint`

### Access

- [https://localhost:5173/](https://localhost:5173/) - Demo Scene
- [https://localhost:5173/benchmark.html](https://localhost:5173/benchmark.html) - Benchmark Setup

### Code Style

There are magic comments, using the form of: `CODE#ABC`. These comments are references from the master thesis to the implementation.

### Publish

1. `git stash --include-untracked`
1. `npm run build`
1. `npm version prerelease --preid=alpha`
1. `npm publish --dry-run`
1. `npm publish`

### Tooling Setup

For working with `wgsl`, it is recommended to use Visual Studio Code with [WGSL Literal](https://marketplace.visualstudio.com/items?itemName=ggsimm.wgsl-literal) extension.
