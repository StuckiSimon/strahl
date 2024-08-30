# strahl 💫

[![Test Pipeline](https://github.com/StuckiSimon/strahl/actions/workflows/test-pipeline.yml/badge.svg)](https://github.com/StuckiSimon/strahl/actions/workflows/test-pipeline.yml)
[![Deploy Website](https://github.com/StuckiSimon/strahl/actions/workflows/deploy-website.yml/badge.svg)](https://github.com/StuckiSimon/strahl/actions/workflows/deploy-website.yml)
[![Build Report](https://github.com/StuckiSimon/strahl/actions/workflows/build-report.yml/badge.svg)](https://github.com/StuckiSimon/strahl/actions/workflows/build-report.yml)

## What is strahl?

**tl;dr** `strahl` is a WebGPU Path Tracer for the web.

strahl is a path tracing library for web applications. Path tracing is a technique based on ray tracing to render 3D models to images with realistic reflections and ambient occlusion without the need for pregenerated artifacts. strahl leverages WebGPU for optimal performance and is based on the OpenPBR surface shading model.

## Quick Links

- [website](https://stuckisimon.github.io/strahl/) — documentation, demo, tutorial, and more information
- [arXiv short paper](https://arxiv.org/abs/2407.19977) — summary of the work
- [report](https://github.com/StuckiSimon/strahl/blob/report/report.pdf) — master thesis report with full details
- [npm package](https://www.npmjs.com/package/strahl) — installable `strahl` package

## Support

Feedback, questions, bug reports, and suggestions are welcome and encouraged. Please contact me directly via one of the following channels:

1. https://x.com/StuckiSimon
1. Mail (see Git commits)
1. https://www.linkedin.com/in/stuckisimon/

## Development

Instructions for working on the library, website, or report of strahl.

### System Requirements

1. Node.js, for version see `.nvmrc`.
1. Browser supporting WebGPU, e.g. Chrome.

### Project Structure

The repo is structured as a monorepo of independent packages. It does not leverage `npm workspaces` or similar. Each project folder is independent and has dedicated documentation.

The packages are:

- [`strahl-lib`](./strahl-lib/README.md) – strahl npm package, the core of the path tracer
- [`website`](./website/README.md) – public website, docs and information
- [`report`](./report/README.md) – LaTeX report of the master thesis

### Setup

`npm ci`

> [!NOTE]
> The workspaces `strahl-lib` and `website` also require to run `npm ci` when first setting them up.

### Prettier

Prettier is used to format files in `strahl-lib` and `website`.

`npm run prettier`

To fix formatting, run:

`npm run prettier:fix`
