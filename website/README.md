# Website

Documentation for strahl built using [Docusaurus](https://docusaurus.io/).

### Installation

`npm ci`

### Local Development

`npm run dev`

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

### Build

`npm run build`

This command generates static content into the `build` directory and can be served using any static contents hosting service.

### Deployment

There is a GitHub Action configured, which automatically deploys the website on changes. Therefore, there is no need to do manual deployment.

Using SSH:

```
$ USE_SSH=true npm run deploy
```

Not using SSH:

```
$ GIT_USER=<Your GitHub username> npm run deploy
```

If you are using GitHub pages for hosting, this command is a convenient way to build the website and push to the `gh-pages` branch.
