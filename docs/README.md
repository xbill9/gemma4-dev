# `docs/` — Medium import sources

Served by GitHub Pages so Medium's importer can fetch them. **Not documentation**
— the rig docs live in each rig directory.

Each page is content-addressed (`<slug>-<sha10>.html`) because **Medium's
importer caches by URL and ignores the query string**: `?v=2` does not bust it.
A changed article must arrive at a URL Medium has never seen, so the filename
carries a hash of the page.

Two other rules are baked into these files and are invisible if you only read
the rendered output:

- **No `<link rel="canonical">`.** The importer resolves it and serves *that*
  URL's cached copy instead of the one you submitted.
- **No links inside `<figcaption>`.** Medium drops the entire figure, silently.

Regenerate with `build_medium.py` rather than editing these by hand.
