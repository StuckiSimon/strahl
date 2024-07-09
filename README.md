# Strahl

WebGPU Ray Tracer

## System Requirements

Node.js, for version see `.nvmrc`.

Browser supporting WebGPU, e.g. Chrome.

## Project Structure

The repo is structured as a monorepo of independent packages. It does not leverage `npm workspaces` or similar. Therefore, each project folder is independent and has dedicated documentation.

The packages are:

- [`strahl-lib`](./strahl-lib/README.md) – strahl npm package, the core of the path tracer
- [`report`](./report/README.md) – LaTeX report of the master thesis

## Setup

`npm ci`
