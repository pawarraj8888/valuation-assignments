# Valuation Assignments

Mini-projects for **FRE 6103 Valuation for Financial Engineering** (NYU Tandon, Prof. David Shimko, Fall 2026),
one folder per assignment.

**Site:** https://pawarraj8888.github.io/valuation-assignments/

| Mini-project | Folder | Dashboard | Authors |
|---|---|---|---|
| 3 (Part 1). Simplified CDO Analysis | [`miniproject-3-cdo-analysis`](miniproject-3-cdo-analysis) | https://pawarraj8888.github.io/valuation-assignments/cdo-analysis/ | Raj Pawar and Michael Brick |
| 2. Creating a ZCB Term Structure | [`miniproject-2-zcb-term-structure`](miniproject-2-zcb-term-structure) | https://pawarraj8888.github.io/valuation-assignments/zcb-term-structure/ | Monalisa Maity and Raj Pawar |
| 1. The Two-Loan Comparison | [`miniproject-1-two-loan-comparison`](miniproject-1-two-loan-comparison) | Excel workbook only | Raj Pawar |

Each folder is self-contained: its own README, data, Python package and tests, notebook, Excel workbook,
outputs and write-up. The commands in a folder's README are run from inside that folder.

## Layout

```
miniproject-1-two-loan-comparison/   Excel workbook
miniproject-2-zcb-term-structure/    data, python, notebooks, excel, output, writeup, docs
miniproject-3-cdo-analysis/          data, python, notebooks, excel, output, writeup, docs
site/                                landing page of the site
.github/workflows/pages.yml          publishes the site
```

## How the site is published

On every push to `main` a GitHub Actions workflow copies `site/` and each mini-project's `docs/` folder into
one site: the landing page at the root, Miniproject 2 under `/zcb-term-structure/` and Miniproject 3 under
`/cdo-analysis/`. Nothing is built on the server. The dashboards are plain HTML and JavaScript that are
generated locally by each project's `build_site.py`.

Miniprojects 2 and 3 used to live in their own repositories (`zcb-term-structure` and `cdo-analysis`). Their
history was brought over with them, and their old dashboard links forward to the pages above.

## License

Code is MIT licensed. Each mini-project folder says what data it includes.
