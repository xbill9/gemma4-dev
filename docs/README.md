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

## Two more things, measured 2026-08-28

**The importer emits an empty code block after every code block, and it does not
matter.** 19 blocks arrive as 38 in the EDITOR, alternating real and empty. Three
fixes were tried and all three were wrong: stripping the trailing newline inside
`<code>`, flattening `<pre><code>` to a bare `<pre>`, and separating adjacent
blocks. It is inherent importer behaviour.

**The empties do not render publicly.** Verified by publishing a throwaway story
with two code blocks and reading the public page: exactly two `<pre>`, zero
empty. So they are an editor-only artifact and need no work at all — do not
spend an afternoon deleting them by hand.

**Medium allows two published-or-scheduled stories per 24 hours.** The throwaway
probe above consumed one of the two slots and delayed the second real article by
a day. If a probe is needed, budget for it or run it on a day with nothing to
ship.
