---
name: Chess Analysis Salon
colors:
  surface: '#111316'
  surface-dim: '#111316'
  surface-bright: '#37393d'
  surface-container-lowest: '#0c0e11'
  surface-container-low: '#1a1c1f'
  surface-container: '#1e2023'
  surface-container-high: '#282a2d'
  surface-container-highest: '#333538'
  on-surface: '#e2e2e6'
  on-surface-variant: '#d0c5af'
  inverse-surface: '#e2e2e6'
  inverse-on-surface: '#2f3034'
  outline: '#99907c'
  outline-variant: '#4d4635'
  surface-tint: '#e9c349'
  primary: '#f2ca50'
  on-primary: '#3c2f00'
  primary-container: '#d4af37'
  on-primary-container: '#554300'
  inverse-primary: '#735c00'
  secondary: '#c3c6ce'
  on-secondary: '#2d3137'
  secondary-container: '#43474e'
  on-secondary-container: '#b2b5bd'
  tertiary: '#6de698'
  on-tertiary: '#00391c'
  tertiary-container: '#4fc97f'
  on-tertiary-container: '#00502a'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#ffe088'
  primary-fixed-dim: '#e9c349'
  on-primary-fixed: '#241a00'
  on-primary-fixed-variant: '#574500'
  secondary-fixed: '#dfe2eb'
  secondary-fixed-dim: '#c3c6ce'
  on-secondary-fixed: '#181c22'
  on-secondary-fixed-variant: '#43474e'
  tertiary-fixed: '#82faab'
  tertiary-fixed-dim: '#64dd91'
  on-tertiary-fixed: '#00210e'
  on-tertiary-fixed-variant: '#00522b'
  background: '#111316'
  on-background: '#e2e2e6'
  surface-variant: '#333538'
typography:
  display-lg:
    fontFamily: EB Garamond
    fontSize: 3.5rem
    fontWeight: '500'
    lineHeight: 4rem
    letterSpacing: -0.02em
  display-lg-mobile:
    fontFamily: EB Garamond
    fontSize: 2.25rem
    fontWeight: '500'
    lineHeight: 2.75rem
    letterSpacing: -0.01em
  headline-lg:
    fontFamily: EB Garamond
    fontSize: 2.25rem
    fontWeight: '600'
    lineHeight: 2.75rem
    letterSpacing: -0.01em
  headline-lg-mobile:
    fontFamily: EB Garamond
    fontSize: 1.75rem
    fontWeight: '600'
    lineHeight: 2.25rem
  headline-md:
    fontFamily: EB Garamond
    fontSize: 1.5rem
    fontWeight: '500'
    lineHeight: 2rem
  headline-sm:
    fontFamily: EB Garamond
    fontSize: 1.25rem
    fontWeight: '500'
    lineHeight: 1.75rem
  body-lg:
    fontFamily: EB Garamond
    fontSize: 1.125rem
    fontWeight: '400'
    lineHeight: 1.85rem
  body-md:
    fontFamily: Manrope
    fontSize: 0.9375rem
    fontWeight: '400'
    lineHeight: 1.5rem
  body-sm:
    fontFamily: Manrope
    fontSize: 0.8125rem
    fontWeight: '400'
    lineHeight: 1.25rem
  label-md:
    fontFamily: Manrope
    fontSize: 0.8125rem
    fontWeight: '600'
    lineHeight: 1rem
    letterSpacing: 0.04em
  label-sm:
    fontFamily: Manrope
    fontSize: 0.6875rem
    fontWeight: '700'
    lineHeight: 0.875rem
    letterSpacing: 0.06em
  notation-mono:
    fontFamily: Manrope
    fontSize: 0.875rem
    fontWeight: '600'
    lineHeight: 1.25rem
    letterSpacing: 0.02em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  gutter: 1.5rem
  gutter-mobile: 0.75rem
  margin: 2rem
  margin-mobile: 1rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 1rem
  space-lg: 1.5rem
  space-xl: 2.5rem
---

## Brand & Style

The design system establishes an atmospheric, contemplative environment tailored for serious chess students, club players, and connoisseurs dissecting the subtle narratives of chess matches. It bridges the intellectual prestige of an old-world private salon—reminiscent of leather-bound tournament anthologies and hand-carved Staunton pieces—with the razor-sharp precision of modern computational chess engines.

The aesthetic marries classical editorial gravitas with clean, analytical data visualization. The visual tone is scholarly, quiet, and decisive: typography commands authority through literary serifs for narrative analysis, while high-legibility geometric sans-serif handles engine evaluations, centipawn tallies, and coordinate systems. The experience feels bespoke, premium, and focused, banishing sensory clutter to facilitate deep immersion and thoughtful study.

## Colors

The palette draws directly from classic salon materials: oxidized brass, polished boxwood, felt tournament tables, and midnight charcoal boards. 

### Core Palette
- **Primary (`#D4AF37`)**: Burnished gold used for focal highlights, active states, key move evaluations, and critical badges.
- **Secondary (`#1E2228`)**: Deep slate container tone providing subtle architectural separation from dark surfaces without harsh contrasts.
- **Tertiary (`#0F9D58`)**: Deep analytical emerald reserved for pristine play ("Brilliant", "Best", and "Great" move annotations).
- **Neutral (`#121417`)**: Pitch charcoal canvas foundation, evoking an intimate, dimly lit study.

### Functional & Move Classification System
- **Brilliant / Best (`#0F9D58` / `#22C55E`)**: Precision green signaling optimal engine-backed continuations.
- **Book / Good (`#64748B` / `#D4AF37`)**: Neutral slate and soft gold denoting established theoretical paths and solid decisions.
- **Inaccuracy / Mistake (`#EAB308` / `#D97706`)**: Warm burnished amber flagging tactical slips or sub-optimal lines.
- **Blunder / Critical Miss (`#DC2626` / `#EF4444`)**: Decisive carmine red commanding immediate diagnostic attention.

Surfaces scale from `#121417` (Canvas Base) up through `#1A1D22` (Card Tier 1) and `#242930` (Interactive Hover / Elevated Overlays). Hairline borders utilize gold at low opacity (`rgba(212, 175, 55, 0.12)`) to yield a fine, gilded edge against charcoal backdrops.

## Typography

The typography strategy builds deliberate tension between historical prose and modern calculation:

1. **Editorial Prose (`EB Garamond`)**: Used for section headlines and narrative commentary. Long-form annotations read like excerpts from classical chess treatises, lending dignity and depth to tactical evaluations.
2. **System Interface & Metadata (`Manrope`)**: Provides functional legibility for moves, centipawn metrics, ply indices, turn indicators, and control toggles.
3. **Hierarchy Rules**: Commentary cards present analytical overviews in `EB Garamond` body scale, while algebraic move records (e.g., `24. Nf3+ Kh8 25. Qh6!`) rely on `Manrope` with tabular figures to maintain vertical rhythm across multi-column move sheets.

## Layout & Spacing

The layout is structured around an asymmetrical workspace that emphasizes analytical flow:

- **Desktop Workspace (12-Column Grid)**: The board maintains a dedicated 7-column anchor, allowing analysis streams, engine evaluations, and annotated game sheets to occupy the remaining 5 columns.
- **Tablet Layout (8-Column Grid)**: The board locks to a 5-column presence, collapsing secondary engine graphs into stacked tabs beneath the narrative thread.
- **Mobile Reflow (4-Column Grid)**: Vertical single-column stack. The chessboard anchors at the top viewport with move commentary sliding underneath in swipeable analytical cards.
- **Spatial Rhythm**: Interior component padding employs a strict 4px base unit. Move list items use `space-xs` and `space-sm` for compact information density, while editorial commentary panels use `space-lg` to create reading room.

## Elevation & Depth

Visual hierarchy uses tonal surface layering combined with faint gilded edge illumination rather than intense dropshadows:

- **Base Layer (`#121417`)**: The root application foundation.
- **Surface Layer 1 (`#1A1D22`)**: Board margins, commentary panels, and secondary module containers. Outlined by a crisp `1px` stroke of `rgba(255, 255, 255, 0.05)`.
- **Surface Layer 2 (`#22262D`)**: Interactive items, dropdowns, hovered move tokens, and modal dialogs.
- **Active / Accent Elevation**: Key moments, best-move highlights, and selected turn cells receive an interior ambient glow (`0 0 16px rgba(212, 175, 55, 0.08)`) and a delicate `1px` border of `rgba(212, 175, 55, 0.35)`.
- **Shadow Philosophy**: Diffused and deeply attenuated (`0 12px 32px -4px rgba(0, 0, 0, 0.65)`), mimicking low-light library conditions and grounding cards firmly onto the background.

## Shapes

The geometric framework balances tailored refinement with technical crispness using a soft profile (`roundedness: 1`):

- **Micro Elements (Move badges, tags, pills)**: `0.25rem` (4px). Preserves crisp, architectural contours matching geometric chessboard geometry.
- **Containers (Panels, commentary cards, engine readouts)**: `0.5rem` (8px, `rounded-lg`). Softens the perimeter without appearing toy-like or overly rounded.
- **Overlays (Modals, popovers)**: `0.75rem` (12px, `rounded-xl`).
- **Chessboard Square Elements**: Pure `0px` radii to maintain edge alignment across rank and file divisions.

## Components

### Buttons
- **Primary Action**: Finished in warm burnished gold (`#D4AF37`) with dark charcoal text (`#121417`), set in bold `Manrope` uppercase labels. Hover state transitions to soft brass (`#E5C158`).
- **Secondary / Ghost**: Deep charcoal surface (`#1A1D22`) with `1px` hairline border (`rgba(212, 175, 55, 0.25)`) and gold label text. Hover brings surface to `#242930` with border opacity rising to `0.5`.

### Move Quality Badges
- Compact pills styled with `label-sm` font, `4px` radius, and `space-xs` vertical by `space-sm` horizontal padding.
- **Brilliant / Best**: Emerald surface tint (`rgba(15, 157, 88, 0.15)`) with vibrant emerald text (`#22C55E`) and border (`rgba(34, 197, 94, 0.3)`).
- **Inaccuracy / Mistake**: Amber surface tint (`rgba(234, 179, 8, 0.15)`) with warm gold text (`#FACC15`) and border (`rgba(234, 179, 8, 0.3)`).
- **Blunder**: Crimson surface tint (`rgba(220, 38, 38, 0.15)`) with bold red text (`#EF4444`) and border (`rgba(239, 68, 68, 0.3)`).

### Move List & Notation Matrix
- Formatted as a dual-column sheet (White / Black) with fixed tabular numbering. 
- Active move highlighted by an illuminated container with a left border accent (`2px solid #D4AF37`) and subtle gold tint (`rgba(212, 175, 55, 0.08)`).
- Move cells feature instant hover feedback (`#22262D`) with seamless keyboard arrow navigation.

### Editorial Commentary Card
- Features a two-tiered anatomy: 
  - **Meta Header**: Minimal `Manrope` metadata (move status badge, engine differential, win-rate percentage).
  - **Prose Body**: Literary commentary rendered in `EB Garamond`, describing the positional ideas, pawn structures, and strategic hazards behind the current ply.
- Outlined with a fine gold hairline (`rgba(212, 175, 55, 0.12)`) on slate-charcoal (`#1A1D22`).

### Input Fields & Search
- Low-profile charcoal fields (`#15181C`) with subtle inset borders (`rgba(255, 255, 255, 0.08)`).
- Active focus state swaps border to burnished gold (`#D4AF37`) with a faint ambient glow. Monospaced notation inputs for FEN/PGN strings.

### Evaluation Bar
- Slender vertical/horizontal strip positioned flush along the chessboard border. White advantage depicted in pale cream (`#EDE8D0`), Black in rich charcoal (`#181B1F`), with a thin, brass-pinned line marking dead-even equilibrium (`0.00`).