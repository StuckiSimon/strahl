# strahl 🌅

WebGPU Path Tracer for the web.

## What is strahl?

strahl is a path tracing library for web applications. Path tracing is a technique based on ray tracing to render 3D models to images with realistic reflections and ambient occlusion without the need for pregenerated artifacts. strahl leverages WebGPU for optimal performance and is based on OpenPBR surface shading model.

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
