---
version: 1
slug: "web-src-app-tsx"
primary_target: "web/src/App.tsx"
related_targets: ["web/src/index.css"]
---

# Surface: Roster Forecast web app

Scope: the whole web app (web/src). Mode: Operate.

Audience and job: a small group of Diamond Dynasty investors on desktop, between monthly attribute
updates. Two equal jobs: find cards to buy (or sell) before the next update, and look up one card's
forecast and why. Feature parity with TBR's tool set where we have real data: projections, buy
lists, rankings, MLB stats tool, track record, flip calculator.

Constraints: monotone business look, no neon or glow (user, binding). Desktop first, works on phone.
No invented claims, reviews, member counts, or exchange rules.

## Direction contract

THESIS: A sober investment desk for The Show, organised like a brokerage research site: a top
navigation of tools, every number backed by the model's track record. It refuses the gamer
dashboard of neon, glows, and loot-box tiles.

OWN-WORLD: Neutral grey scale on white (system dark mode mirrors it), near-black ink, one ink-blue
for links and focus only; green and red appear only on up/down values. Inter with tabular figures,
hairline rules, 6px radii, no shadows except overlays. Card art is the only imagery.

STORY: The visitor sees how many cards are likely to move, scans a ranked list, opens a card to see
odds, attribute ranges, real stats and a profit calculator, and can check the track record.

FIRST VIEWPORT: Top bar with wordmark and nav (Projections, Buy Lists, Player Stats, Track Record),
next-update date at right. Page title with a one-line forecast summary, then tabs with counts, filters and the
projections table with card thumbnails, starting above the fold.

FORM: Category standard (canon), taken by the user over the dealt hand; seed 7eea8892.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
