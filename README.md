# strahl 🌅

[![Test Pipeline](https://github.com/StuckiSimon/strahl/actions/workflows/test-pipeline.yml/badge.svg)](https://github.com/StuckiSimon/strahl/actions/workflows/test-pipeline.yml)

WebGPU Path Tracer for the web.

> [!NOTE]
> This is still a work in progress, if you're interested in the work, take a look at the related short paper on [arXiv](https://arxiv.org/abs/2407.19977).

## What is strahl?

strahl is a path tracing library for web applications. Path tracing is a technique based on ray tracing to render 3D models to images with realistic reflections and ambient occlusion without the need for pregenerated artifacts. strahl leverages WebGPU for optimal performance and is based on OpenPBR surface shading model.

For more information, take a look at the [website](https://stuckisimon.github.io/strahl/).

## Current State

The public documentation is still a work in progress and is expected to be finished until end of August. While you're here, feel free to checkout the short-paper describing the work on [arXiv](https://arxiv.org/abs/2407.19977).

Feedback and questions are welcomed on any direct channel:

1. https://x.com/StuckiSimon
1. Mail (see Git commits)
1. https://www.linkedin.com/in/stuckisimon/

## System Requirements

Node.js, for version see `.nvmrc`.

Browser supporting WebGPU, e.g. Chrome.

## Project Structure

The repo is structured as a monorepo of independent packages. It does not leverage `npm workspaces` or similar. Therefore, each project folder is independent and has dedicated documentation.

The packages are:

- [`strahl-lib`](./strahl-lib/README.md) – strahl npm package, the core of the path tracer
- [`website`](./website/README.md) – public website, docs and information
- [`report`](./report/README.md) – LaTeX report of the master thesis

## Setup

`npm ci`
