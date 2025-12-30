---
title: "Conventional Commits: Make Team Collaboration and Automation Efficient"
date: 2025-10-25
draft: false
---

### Title

From `feat` to `fix`: master Git commit conventions for collaboration and automation

---

### Subtitle / Abstract

A practical guide to Conventional Commits. Learn commit types (`feat:`, `fix:`), write clean messages, and enable automatic changelogs and releases.

---

### Target readers

- **Beginners**: new to Git, want better commit habits.
- **Mid-level devs**: want commits friendly to team and CI.
- **Leads/architects**: want a consistent team standard.

---

### Background / Motivation

Most commit messages look like:

> "update code"
> "fix bug"
> "some changes"

They are readable short-term but useless long-term. As teams grow, it becomes hard to track intent or automate releases.

**Conventional Commits** provides a simple, unified format so commits are **readable, traceable, and automatable**.

---

### Core concept

Conventional Commits define a commit message structure:

```
<type>(<scope>): <subject>

<body>

<footer>
```

- `type`: commit type, e.g. `feat`, `fix`, `docs`
- `scope`: optional area, e.g. `ui`, `api`
- `subject`: short description (<= 50 chars)
- `body`: details (optional)
- `footer`: metadata (e.g., BREAKING CHANGE)

---

### Practical steps

1) **Set Git editor to Neovim (optional)**

```bash
git config --global core.editor "nvim"
```

2) **Write a standard commit message**

```bash
git commit -m "feat(lsp): support new nvim-lspconfig API"
```

3) **Structured commit example**

```
feat(lsp): update LSP config for new nvim-lspconfig

- remove old lspconfig[server].setup
- use new function call lspconfig(server, {...})
```

4) **Enforce with tooling (optional)**

```bash
npm install -g commitlint @commitlint/config-conventional
```

Create `.commitlintrc.js`:

```js
module.exports = { extends: ["@commitlint/config-conventional"] };
```

---

### Runnable examples

```bash
# new feature
git commit -m "feat(auth): add two-factor login"

# bug fix
git commit -m "fix(ui): fix text invisibility in dark mode"

# docs
git commit -m "docs(readme): add usage notes"

# refactor
git commit -m "refactor(api): optimize auth logic"

# performance
git commit -m "perf(db): improve query cache"
```

---

### Explanation

This standard comes from the **Angular** commit message format and became the **Conventional Commits** spec.

**Benefits:**

- Clear structure: see type and scope at a glance
- Machine-readable: auto changelog generation
- Easy integration: with `semantic-release`

**Alternatives:**

- [Gitmoji](https://gitmoji.dev/) (emoji commits)
- [Semantic Versioning](https://semver.org/) (release versioning)

---

### FAQ

| Question | Answer |
| --- | --- |
| Can I mix English and Chinese? | Yes, but keep it consistent. Prefer English in the subject. |
| What if one commit covers multiple types? | Split into multiple commits. |
| What if I cannot write a long message? | At least explain "why" in one line. |
| Is scope required? | Optional, but recommended. |

---

### Best practices

- One commit does one thing
- Subject in lowercase, no period
- First line <= 50 characters
- Blank line after subject, details start on line 3
- Start with a verb (add, fix, update)

---

### Conclusion

Commit conventions are a small investment with big returns: better history, smoother collaboration, and automation.

---

### References

- [Conventional Commits](https://www.conventionalcommits.org/)
- [Angular Commit Message Guidelines](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit)
- [semantic-release](https://semantic-release.gitbook.io/)
- [Gitmoji](https://gitmoji.dev/)

---

### Meta

- Reading time: about 6 minutes
- Tags: Git, standards, Conventional Commits, collaboration
- SEO keywords: Git commit conventions, Conventional Commits, feat fix refactor, best practices
- Meta description: A practical guide to writing clean commit messages with `feat:` and `fix:` and enabling automation.

---

### Call to Action (CTA)

Try this in your next commit:

```bash
git commit -m "feat: first commit with conventional commits"
```

Share your team conventions and lessons learned in the comments.
