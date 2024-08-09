# strahl-lib

Main library package provided as [NPM package](https://www.npmjs.com/package/strahl).

## Development

`npm run dev`

In order to expose in local network, run:

`npm run dev -- --host`

## Publish

1. `git stash --include-untracked`
1. `npm run build`
1. `npm version prerelease --preid=alpha`
1. `npm publish --dry-run`
1. `npm publish`
