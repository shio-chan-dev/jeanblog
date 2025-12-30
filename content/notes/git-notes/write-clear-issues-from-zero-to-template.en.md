---
title: "Write Clear Issues with Templates: From Zero to GitHub Issue Forms"
date: 2025-11-11
draft: false
---

## Title (accurate and keyword-rich)

**Write Clear Requirements with Issue Templates: A Complete Guide to GitHub Issue Forms**

---

## Subtitle / Abstract

This post teaches you how to configure GitHub Issue templates for feature requests and bug reports, including folder structure, YAML forms, Markdown templates, and common pitfalls. It is ideal for teams that want clearer requirements and less back-and-forth.

---

## Target readers

This article is for:

- Backend/frontend/full-stack engineers who create Issues regularly
- Leads/TLs/architects who want standardized requirement intake
- Mid-level developers familiar with GitHub but new to Issue templates

Beginners can follow, but basic GitHub knowledge is assumed.

---

## Background / Motivation: Why templates?

Without templates, you often hear:

- "What is the background?"
- "Which modules are affected?"
- "What is the acceptance criteria?"
- "How high is the priority?"

A one-line Issue like:

> "Add export feature"

will confuse everyone.

Long-term pain points:

1. **High communication cost**: details must be asked repeatedly
2. **Information asymmetry**: the requester knows, but the Issue does not
3. **Hard to plan**: no priority or acceptance criteria
4. **Hard to trace**: months later, nobody knows the intent

GitHub Issue Templates are structured questions that guide good input:

- enforce or guide required fields
- auto-label and auto-prefix titles
- support form UI validation

Goal: make every new Issue understandable at a glance.

---

## Core concepts

### 1. Issue Template

- A preset format shown when creating a new Issue
- Can be Markdown text or YAML form

### 2. Markdown template

- Old and simple
- Essentially a prefilled Markdown file
- Path: `.github/ISSUE_TEMPLATE/xxx.md` or `.github/ISSUE_TEMPLATE.md`

### 3. YAML Issue form

- New and recommended
- Form UI with inputs and dropdowns
- Submissions are converted to Markdown in the Issue body
- Path: `.github/ISSUE_TEMPLATE/xxx.yml`

### 4. config.yml

- Path: `.github/ISSUE_TEMPLATE/config.yml`
- Controls:
  - Whether blank Issues are allowed
  - Which templates are displayed

---

## Practice guide / Steps overview

1. Create `.github/ISSUE_TEMPLATE`
2. Create a Feature template (YAML form)
3. (Optional) Create a Bug template
4. Configure `config.yml` for blank Issue behavior
5. Commit and push to GitHub
6. Verify in the GitHub UI

---

## Step 1: Create the template folder

```bash
mkdir -p .github/ISSUE_TEMPLATE
```

Structure:

```text
your-repo/
  .github/
    ISSUE_TEMPLATE/
      # yml / md files go here
  src/
  ...
```

---

## Step 2: Feature request template (YAML form)

Create `.github/ISSUE_TEMPLATE/feature-request.yml`:

```yaml
name: "Feature Request"
description: "Use for new features or requirement changes"
title: "[Feature] "
labels:
  - "feature"
  - "enhancement"

body:
  - type: markdown
    attributes:
      value: |
        Thanks for submitting a feature request.
        Please be as clear as possible for evaluation and planning.

  - type: input
    id: module
    attributes:
      label: Affected module
      description: Service or module involved (API, crawler, UI, etc.)
      placeholder: e.g. export API / attachment viewer
    validations:
      required: true

  - type: textarea
    id: background
    attributes:
      label: Background / scenario
      description: Why do we need this? What problem are we solving?
      placeholder: |
        Describe business context, roles, usage scenario, pain points...
    validations:
      required: true

  - type: textarea
    id: description
    attributes:
      label: Requirement description
      description: Describe desired behavior from the user perspective.
      placeholder: |
        1. Add ... on page ...
        2. When user does ..., the system should ...
        3. Edge cases to support: ...
    validations:
      required: true

  - type: textarea
    id: acceptance_criteria
    attributes:
      label: Acceptance criteria
      description: What counts as "done"? Helps testing and review.
      placeholder: |
        - [ ] Scenario 1: ...
        - [ ] Scenario 2: ...
        - [ ] Performance / security requirements: ...
    validations:
      required: true

  - type: dropdown
    id: priority
    attributes:
      label: Priority
      description: For planning and ordering
      options:
        - P0 (must be done this iteration)
        - P1 (high)
        - P2 (normal)
        - P3 (low)
      default: 2
    validations:
      required: false

  - type: textarea
    id: extra
    attributes:
      label: Extra info
      description: Related APIs, docs, designs, screenshots, linked issues
      placeholder: |
        - API docs:
        - Design / prototype:
        - Related Issue / ticket:
    validations:
      required: false
```

**Effect:**

- A "Feature Request" option appears on Issue creation
- The form replaces plain text
- Labels `feature` and `enhancement` are added
- Title is prefixed with `[Feature] `
- Key fields are required

---

## Step 3 (Optional): Bug report template

Create `.github/ISSUE_TEMPLATE/bug-report.yml`:

```yaml
name: "Bug Report"
description: "Use for bugs and exceptions"
title: "[Bug] "
labels:
  - "bug"

body:
  - type: textarea
    id: summary
    attributes:
      label: Summary
      placeholder: Briefly describe the problem
    validations:
      required: true

  - type: textarea
    id: steps
    attributes:
      label: Steps to reproduce
      placeholder: |
        1. Open ...
        2. Click ...
        3. See ...
    validations:
      required: true

  - type: textarea
    id: expected
    attributes:
      label: Expected result
    validations:
      required: true

  - type: textarea
    id: actual
    attributes:
      label: Actual result
    validations:
      required: true

  - type: textarea
    id: extra
    attributes:
      label: Extra info
      description: Logs, screenshots, environment details
    validations:
      required: false
```

Now your team can separate feature requests from bugs clearly.

---

## Step 4: Configure `config.yml`

Create `.github/ISSUE_TEMPLATE/config.yml`:

```yaml
blank_issues_enabled: false  # disallow blank Issues, force templates
contact_links:
  - name: Internal request system
    url: https://example.com/your-internal-system
    about: If this is a formal request, create it in the internal system first.
```

If you do not have an internal system, remove `contact_links` or replace with your wiki.

`blank_issues_enabled: false` forces template usage and prevents empty Issues.

---

## Step 5: Commit and push

```bash
git add .github/ISSUE_TEMPLATE/*
git commit -m "chore: add GitHub issue templates for feature and bug"
git push
```

Templates take effect on the default branch (usually `main` or `master`).

---

## Step 6: Verify in GitHub UI

1. Open your repo
2. Click **Issues**
3. Click **New issue**

You should see template choices such as:

- Feature Request
- Bug Report
- (Optional) Open a blank issue

If `blank_issues_enabled: false`, the blank option disappears.

---

## Minimal runnable example

If you only need a minimal Feature template, do this:

**1) Create folder:**

```bash
mkdir -p .github/ISSUE_TEMPLATE
```

**2) Create `.github/ISSUE_TEMPLATE/feature-request.yml`:**

```yaml
name: "Feature Request"
description: "Use for new features or requirement changes"
title: "[Feature] "
labels: ["feature"]

body:
  - type: textarea
    id: background
    attributes:
      label: Background / scenario
      placeholder: Briefly describe why this is needed
    validations:
      required: true

  - type: textarea
    id: description
    attributes:
      label: Requirement description
      placeholder: |
        What should the system do? How will users use it?
    validations:
      required: true

  - type: textarea
    id: acceptance
    attributes:
      label: Acceptance criteria
      placeholder: |
        - [ ] Scenario 1: ...
        - [ ] Scenario 2: ...
    validations:
      required: true
```

Then:

```bash
git add .github/ISSUE_TEMPLATE/feature-request.yml
git commit -m "add minimal feature request issue template"
git push
```

---

## Why YAML forms instead of Markdown?

### Benefits of YAML forms

- **Required field validation** for background/requirements/acceptance
- **Friendly UI** for non-technical teammates
- **Clear structure** for reading and automation
- **Auto labels and title prefixes**

### Markdown template pros and cons

Markdown templates are fine, but:

- Pros:
  - simple and compatible
  - good for technical teams
- Cons:
  - cannot enforce required fields
  - UI is less friendly for product/ops roles

If you want **team standards and clarity**, YAML forms are better. For small personal projects, Markdown is enough.

---

## Common issues and notes

### 1. Template not working?

Check:

- correct path: `.github/ISSUE_TEMPLATE/xxx.yml` or `.github/ISSUE_TEMPLATE/xxx.md`
- default branch: template must be on `main`/`master`
- case sensitivity: `ISSUE_TEMPLATE` must match exactly

### 2. Template updated but UI unchanged?

- browser cache: refresh or use private mode
- confirm you pushed to GitHub
- for forks, templates are per-repo and do not inherit upstream

### 3. Org-level templates?

You can configure templates in an org-wide `.github` repo. Repos without templates will use the org default.

### 4. YAML errors?

- YAML is sensitive to indentation and spaces
- GitHub may ignore or error on malformed YAML
- Use editor validation (VS Code is great)

---

## Best practices

1. Start simple: one feature template first
2. Require core fields: background, description, acceptance
3. Standardize title prefixes: `[Feature]`, `[Bug]`
4. Auto-label to reduce manual maintenance
5. Keep forms short; balance clarity and friction
6. Review after 1-2 months and refine fields

---

## Conclusion

This guide covers:

- Issue template concepts (YAML forms vs Markdown)
- A full Feature template plus optional Bug template
- The end-to-end workflow: create -> write -> push -> verify
- Why YAML forms are recommended

If you apply these steps, requirement quality will improve immediately.

---

## References and further reading

- GitHub Docs: Issue and pull request templates
- Keywords:
  - `github issue template yaml`
  - `github issue forms`
  - `github .github/ISSUE_TEMPLATE examples`

---

## Meta

- Reading time: 8-12 minutes
- Tags: GitHub, collaboration, Issue templates, team standards, requirements
- SEO keywords:
  - GitHub Issue template
  - GitHub Issue Template config
  - YAML Issue Form tutorial
  - Feature request template
- Meta description:
  A complete guide to configuring GitHub Issue templates (YAML forms and Bug templates) with best practices and pitfalls.

---

## Call to Action (CTA)

If you finished reading, do this now:

1. Pick a GitHub repo you use often
2. Create `.github/ISSUE_TEMPLATE/feature-request.yml`
3. Push and open a test Issue to see the effect
