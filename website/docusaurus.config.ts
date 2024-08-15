import { themes as prismThemes } from "prism-react-renderer";
import type { Config } from "@docusaurus/types";
import type * as Preset from "@docusaurus/preset-classic";

const config: Config = {
  title: "strahl",
  tagline: "WebGPU path tracer using OpenPBR",
  favicon: "img/favicon.ico",

  url: "https://stuckisimon.github.io",
  baseUrl: "/strahl/",

  // GitHub pages deployment config
  organizationName: "StuckiSimon",
  projectName: "strahl",

  onBrokenLinks: "throw",
  onBrokenMarkdownLinks: "warn",

  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },

  plugins: [
    [
      "docusaurus-plugin-typedoc",
      {
        entryPoints: ["../strahl-lib/src/index.ts"],
        tsconfig: "../strahl-lib/tsconfig.json",
        watch: process.env.TYPEDOC_WATCH,
      },
    ],
  ],

  presets: [
    [
      "classic",
      {
        docs: {
          sidebarPath: "./sidebars.ts",
        },
        blog: {
          showReadingTime: true,
        },
        theme: {
          customCss: "./src/css/custom.css",
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    colorMode: {
      defaultMode: "dark",
      disableSwitch: true,
      respectPrefersColorScheme: false,
    },

    navbar: {
      title: "strahl",

      items: [
        {
          type: "docSidebar",
          sidebarId: "tutorialSidebar",
          position: "left",
          label: "Documentation",
        },
        { to: "/blog", label: "Blog", position: "left" },
      ],
    },
    footer: {
      style: "dark",
      links: [
        {
          title: "Quick Navigation",
          items: [
            {
              label: "Documentation",
              to: "/docs/intro",
            },
            {
              label: "Blog",
              to: "/blog",
            },
          ],
        },
        {
          title: "External",
          items: [
            {
              label: "GitHub",
              href: "https://github.com/StuckiSimon/strahl",
            },
            {
              label: "Twitter",
              href: "https://twitter.com/StuckiSimon",
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Simon Stucki. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      magicComments: [
        // Code Diff Higlights
        // See: https://github.com/facebook/docusaurus/issues/3318#issuecomment-1909563681
        {
          className: "code-block-diff-add-line",
          line: "diff-add",
        },
        {
          className: "code-block-diff-remove-line",
          line: "diff-remove",
        },
      ],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
