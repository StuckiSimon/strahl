# report

This is the main master thesis report associated with `strahl`.

## Usage

To compile the report, follow these steps:

1. Install LaTeX distribution (e.g., TeX Live, MiKTeX).
2. Open `main.tex` in your LaTeX editor.
3. Compile the document using the appropriate LaTeX command.
4. The compiled report will be generated as `main.pdf`.

### Publishing

No manual publishing or building is required as the report is automatically built using GitHub Actions. This ensures that automatic control of the report is done and makes distribution of the report straightforward.

### Visual Studio Code Configuration

The [LaTeX Workshop Extension](https://github.com/James-Yu/LaTeX-Workshop) for Visual Studio Code is recommended for development.

The build command can be configured in the settings for the key `latex-workshop.latex.tools` as:

```json
{
  "name": "latexmk",
  "command": "latexmk",
  "args": [
    "-synctex=1",
    "-interaction=nonstopmode",
    "-file-line-error",
    "-pdf",
    "-pdflatex=pdflatex -shell-escape %O %S",
    "-outdir=%OUTDIR%",
    "%DOC%"
  ]
},
```

Note the use of `-shell-escape` which is required for `\tikzexternalize`.
