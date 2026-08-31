#!/usr/bin/env node
/**
 * Manual Algolia indexer for the built Docusaurus site.
 *
 * Walks docs-site/build/**\/index.html, extracts one record per
 * heading section (h1/h2/h3 + the text under it), and pushes them
 * to the `veloxquant-mlx` index via `saveObjects` (clears the index
 * first so removed/renamed pages don't leave stale records behind).
 *
 * Usage:
 *   ALGOLIA_APP_ID=... ALGOLIA_ADMIN_API_KEY=... node scripts/algolia-index.js
 *
 * Requires an Admin API key (write access) — the search-only key used
 * by the site itself cannot write to the index.
 */
const fs = require('fs');
const path = require('path');
const cheerio = require('cheerio');
const { algoliasearch } = require('algoliasearch');

const BUILD_DIR = path.join(__dirname, '..', 'build');
const INDEX_NAME = 'veloxquant-mlx';
const SITE_URL = 'https://veloxquant-mlx.netlify.app';

const SKIP_PATH_SEGMENTS = ['assets', 'img', '404'];
// Algolia caps records at 10KB; stay well under that since objectID/url/
// hierarchy fields also count toward the total.
const MAX_CONTENT_CHARS = 4000;

function findHtmlFiles(dir) {
  const results = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (SKIP_PATH_SEGMENTS.includes(entry.name)) continue;
      results.push(...findHtmlFiles(full));
    } else if (entry.name === 'index.html') {
      results.push(full);
    }
  }
  return results;
}

function urlFor(htmlFile) {
  const rel = path.relative(BUILD_DIR, path.dirname(htmlFile));
  const urlPath = rel === '.' ? '/' : `/${rel}/`;
  return `${SITE_URL}/docs${urlPath === '/' ? '' : urlPath}`;
}

function extractRecords(htmlFile) {
  const html = fs.readFileSync(htmlFile, 'utf8');
  const $ = cheerio.load(html);

  const article = $('article').first();
  if (article.length === 0) return [];

  const pageTitle = $('article h1').first().text().trim() || $('title').text().trim();
  const url = urlFor(htmlFile);

  const records = [];
  let current = null;
  let hierarchy = { lvl0: pageTitle, lvl1: null, lvl2: null, lvl3: null };

  const flush = () => {
    if (current && current.content.trim().length > 0) {
      records.push(current);
    }
  };

  article
    .find('h1, h2, h3, p, li')
    .each((_, el) => {
      const tag = el.tagName.toLowerCase();
      const text = $(el).text().replace(/​/g, '').trim();
      if (!text) return;

      if (tag === 'h1' || tag === 'h2' || tag === 'h3') {
        flush();
        if (tag === 'h1') {
          hierarchy = { lvl0: text, lvl1: null, lvl2: null, lvl3: null };
        } else if (tag === 'h2') {
          hierarchy = { ...hierarchy, lvl1: text, lvl2: null, lvl3: null };
        } else {
          hierarchy = { ...hierarchy, lvl2: text, lvl3: null };
        }
        current = {
          objectID: `${url}#${text}`,
          url: hierarchy.lvl2
            ? `${url}#${slugify(hierarchy.lvl2)}`
            : hierarchy.lvl1
            ? `${url}#${slugify(hierarchy.lvl1)}`
            : url,
          pageTitle,
          hierarchy: { ...hierarchy },
          content: '',
        };
      } else if (current) {
        if (current.content.length < MAX_CONTENT_CHARS) {
          current.content += (current.content ? ' ' : '') + text;
        }
      }
    });
  flush();

  // Always index at least the page itself, even if it has no sub-headings.
  if (records.length === 0) {
    records.push({
      objectID: url,
      url,
      pageTitle,
      hierarchy: { lvl0: pageTitle, lvl1: null, lvl2: null, lvl3: null },
      content: article.text().replace(/\s+/g, ' ').trim().slice(0, 2000),
    });
  }

  return records;
}

function slugify(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/(^-|-$)/g, '');
}

async function main() {
  const appId = process.env.ALGOLIA_APP_ID;
  const adminKey = process.env.ALGOLIA_ADMIN_API_KEY;

  if (!appId || !adminKey) {
    console.error(
      'Missing ALGOLIA_APP_ID or ALGOLIA_ADMIN_API_KEY env vars. ' +
        'The admin key is required to write to the index (search-only keys cannot write).'
    );
    process.exit(1);
  }

  if (!fs.existsSync(BUILD_DIR)) {
    console.error(`Build directory not found at ${BUILD_DIR}. Run "npm run build" first.`);
    process.exit(1);
  }

  const htmlFiles = findHtmlFiles(BUILD_DIR);
  console.log(`Found ${htmlFiles.length} pages under ${BUILD_DIR}`);

  const records = htmlFiles.flatMap(extractRecords);
  console.log(`Extracted ${records.length} records`);

  const client = algoliasearch(appId, adminKey);

  console.log(`Clearing existing objects in "${INDEX_NAME}"...`);
  await client.clearObjects({ indexName: INDEX_NAME });

  console.log(`Pushing ${records.length} records to "${INDEX_NAME}"...`);
  await client.saveObjects({ indexName: INDEX_NAME, objects: records });

  await client.setSettings({
    indexName: INDEX_NAME,
    indexSettings: {
      searchableAttributes: ['pageTitle', 'hierarchy.lvl1', 'hierarchy.lvl2', 'content'],
      attributesToSnippet: ['content:20'],
    },
  });

  console.log('Done.');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
