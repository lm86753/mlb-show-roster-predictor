# Design

The dashboard (`web/`) is a sober research desk for Diamond Dynasty investors, laid out like a
brokerage research site: a top navigation of tools, every number backed by the model's track
record. It deliberately avoids the gamer look (neon, glows, loot-box tiles).

## Principles

- **Monotone.** Neutral greys on white, near-black ink. Dark mode follows the system and mirrors
  the light palette. Colour carries meaning only: green/red for up/down values, amber for rarity
  tier jumps, one ink-blue for links and focus.
- **Numbers first.** Inter with tabular figures, right-aligned numeric columns, hairline rules,
  6px radii. No shadows except on overlays (drawer, menus).
- **Card art is the only imagery.** Thumbnails come from The Show's CDN; no decorative graphics.
- **No invented claims.** Every figure on the page comes from the model output or the backtest.
  Empty data windows (e.g. last-30-day stats after the regular season) are hidden, not shown as
  rows of dashes.

## Tokens

All colours, the font stack, radius and page width are CSS custom properties at the top of
`web/src/index.css` (`--bg`, `--surface`, `--text`, `--text-2/3`, `--border`, `--ink`, `--link`,
`--up`, `--down`, `--tier`, …), redefined under `prefers-color-scheme: dark`. Components use the
tokens only; no raw hex values elsewhere.

## Structure

| Page | Purpose |
|---|---|
| Projections | One-line forecast summary, tabs (likely upgrades, likely downgrades, quicksell value, all cards), filters, table or card grid |
| Buy Lists | Short ranked lists with the rule behind each list |
| Player Stats | Real 2026 MLB numbers next to each card's rating and forecast |
| Track Record | Walk-forward backtest results and the latest update's predicted-vs-actual picks |
| Player drawer | Outcome odds, MLB stats, attribute projections with 80% ranges and a scenario editor, flip calculator, update history |

Routing is hash-based (`web/src/routes.ts`) so the static build deploys anywhere.

## Responsive

Desktop first (1240px page), checked at 1440px and 390px. Below 640px the side gutter drops to
16px, secondary columns (`.hide-sm`) are hidden, wide tables scroll inside their own container,
and the page itself never scrolls horizontally.
