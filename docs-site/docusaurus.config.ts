import { themes as prismThemes } from 'prism-react-renderer';
import type { Config } from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'VeloxQuant-MLX',
  tagline: '16× KV cache compression for Apple Silicon',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://veloxquant-mlx.netlify.app',
  baseUrl: '/docs/',

  onBrokenLinks: 'throw',
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          routeBasePath: '/',
          sidebarPath: './sidebars.ts',
          editUrl:
            'https://github.com/rajveer43/veloxquant-mlx/edit/master/docs-site/',
        },
        blog: {
          showReadingTime: true,
          blogTitle: 'VeloxQuant-MLX Blog',
          blogDescription: 'Deep dives into KV cache quantization, Metal kernels, and Apple Silicon LLM inference',
          postsPerPage: 'ALL',
          blogSidebarTitle: 'All posts',
          blogSidebarCount: 'ALL',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
        sitemap: {
          lastmod: 'date',
          changefreq: null,
          priority: null,
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    colorMode: {
      defaultMode: 'dark',
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },
    image: 'img/favicon.ico',
    navbar: {
      title: 'VeloxQuant-MLX',
      logo: {
        alt: 'VeloxQuant-MLX Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'gettingStartedSidebar',
          position: 'left',
          label: 'Getting Started',
        },
        {
          type: 'docSidebar',
          sidebarId: 'algorithmsSidebar',
          position: 'left',
          label: 'Algorithms',
        },
        {
          type: 'docSidebar',
          sidebarId: 'guidesSidebar',
          position: 'left',
          label: 'Guides',
        },
        {
          type: 'docSidebar',
          sidebarId: 'apiSidebar',
          position: 'left',
          label: 'API Reference',
        },
        {
          to: '/changelog',
          label: 'Changelog',
          position: 'left',
        },
        {
          to: '/blog',
          label: 'Blog',
          position: 'left',
        },
        {
          href: 'https://github.com/rajveer43/veloxquant-mlx',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            { label: 'Getting Started', to: '/getting-started/intro' },
            { label: 'Algorithms', to: '/algorithms/overview' },
            { label: 'Guides', to: '/guides/mlx-lm-integration' },
            { label: 'API Reference', to: '/api/cache' },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/rajveer43/veloxquant-mlx',
            },
            {
              label: 'Issues',
              href: 'https://github.com/rajveer43/veloxquant-mlx/issues',
            },
            {
              label: '☕ Buy me a coffee',
              href: 'https://buymeacoffee.com/rajveer43',
            },
          ],
        },
        {
          title: 'More',
          items: [
            { label: 'Blog', to: '/blog' },
            { label: 'Changelog', to: '/changelog' },
            { label: 'Home', href: '/' },
          ],
        },
      ],
      copyright: `MIT License © ${new Date().getFullYear()} VeloxQuant-MLX. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python', 'bash', 'toml'],
    },
    // algolia search is configured via env vars on Netlify; disabled locally
    ...(process.env.ALGOLIA_APP_ID ? {
      algolia: {
        appId: process.env.ALGOLIA_APP_ID!,
        apiKey: process.env.ALGOLIA_SEARCH_API_KEY!,
        indexName: 'veloxquant-mlx',
        // contextualSearch requires facets (lang/docusaurus_tag) that only
        // the official DocSearch crawler writes. This index is built by our
        // own scripts/algolia-index.js, which doesn't set those facets, so
        // contextualSearch's auto-applied filter matches zero records and
        // every query silently returns no results. Leave it off until the
        // indexer writes matching facets.
        contextualSearch: false,
      },
    } : {}),
  } satisfies Preset.ThemeConfig,
};

export default config;
