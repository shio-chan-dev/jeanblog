# Layout Overrides

This project overrides the PaperMod header menu generation to avoid duplicated
language prefixes (for example, `/zh/jeanblog/zh/...`) after switching to
language-specific content directories.

Changes:
- `layouts/partials/header.html` is copied from the PaperMod module and modified.
- Menu links use the menu entry's `.Page.RelPermalink` when available.
- External URLs are left untouched.

Why:
- `absLangURL` on `.URL` can duplicate the language subpath when the menu is
  generated inside the language root and `baseURL` already includes a subpath.

If you update the theme module, review and re-apply this override as needed.
