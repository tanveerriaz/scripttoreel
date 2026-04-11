# ScriptToReel branding (Gemini spec)

## Color tokens

| Name | Hex | RGB | Purpose |
|------|-----|-----|---------|
| Luminous Lime | `#CCFF00` | (204, 255, 0) | Accent: icon symbol, “Reel” wordmark, UI accent |
| Deep Charcoal | `#121212` | (18, 18, 18) | App icon container (squircle field) |
| UI Black | `#1C1C1E` | (28, 28, 30) | Alternative dark UI surfaces |
| Standard Black | `#000000` | (0, 0, 0) | Primary text (“ScriptTo” on light) |
| Pure White | `#FFFFFF` | (255, 255, 255) | Reverse / monochrome light base |

Product dashboard ([`dashboard.html`](../dashboard.html)) uses **Luminous Lime** as Tailwind `accent` for buttons and highlights.

## Dynamic Motion Transition symbol

- **Canvas:** Design at **512×512** (mark) or **1024×1024** (app icon).
- **Motion lines (left):** Exactly **five** horizontal strokes, **fully rounded** caps (`rx = height/2`). **Shortest at the top**, **longest at the bottom** (acceleration left → right).
- **Film reel (right):** Classic strip with **three frame windows**; **sprocket holes** as small rounded rects along **top and bottom** edges.
- **Transition:** A **bridge** shape merges the cluster into the film block; the **longest** motion line aligns into the film structure.

## App icon container

- **Fill:** Deep Charcoal `#121212`.
- **Corner radius:** **22.5%** of side length (e.g. 1024 → `rx="230"`).
- **Symbol scale:** Merged mark scaled to **~67%** of icon height (`scale(686/512) ≈ 1.33984375`), **centered** (`translate(169, 169)` on 1024 canvas).
- **Clear space:** **10%** margin on the full 1024 artboard (102px). Reference: [`guidelines/clear-space-spec.svg`](guidelines/clear-space-spec.svg).

## Wordmark (stacked)

- **Font:** Geometric sans, **Inter** Bold (700) (alternatives: Montserrat, Poppins).
- **Case:** PascalCase **ScriptToReel** (cap **S**, **T**, **R**).
- **Colors:** `ScriptTo` → `#000000`; `Reel` → `#CCFF00` (on light backgrounds).
- **Layout:** See [`logo-stacked.svg`](logo-stacked.svg) — canvas **1536×1200**, icon width 1024 centered; wordmark sized ~**1.5×** icon width; vertical gap from icon bottom to text baseline ≈ **0.5 × font-size**.

## SVG files

| File | Notes |
|------|--------|
| [`mark.svg`](mark.svg) | Transparent; Luminous Lime `#CCFF00` only. |
| [`icon-1024.svg`](icon-1024.svg) / [`app-icon.svg`](app-icon.svg) | `#121212` tile, `rx=230`, scaled mark. |
| [`logo-stacked.svg`](logo-stacked.svg) | White canvas, spec wordmark colors. |
| [`icon-reverse.svg`](icon-reverse.svg) | `#FFFFFF` tile, lime mark. |
| [`icon-mono-dark.svg`](icon-mono-dark.svg) | White mark on `#121212`. |
| [`icon-mono-light.svg`](icon-mono-light.svg) | `#000000` mark on `#FFFFFF`. |
| [`app-icon-legacy.svg`](app-icon-legacy.svg) | Old Wikimedia reel (reference). |
| [`scripttoreel-logo.svg`](scripttoreel-logo.svg) | **Primary marketing lockup** from Gemini (large file): embedded bitmap + `feColorMatrix` filters—use this SVG as a whole, not a PNG extract. |
| [`logo-lockup.svg`](logo-lockup.svg) | Legacy thin wrapper + PNG (sidebar no longer uses this). |
| [`logo-lockup-1024.png`](logo-lockup-1024.png) | Web-sized lockup (max 1024px wide). |

## Source raster

Gemini sheet copy: [`source/logo-spec-sheet.png`](source/logo-spec-sheet.png). Crop math: [`source/CROPS.md`](source/CROPS.md).

Updated full-frame lockup from Gemini (extracted from the “SVG” file): [`source/logo-lockup-from-gemini.png`](source/logo-lockup-from-gemini.png) (2816×1536).

## Regenerating guideline PNGs

From repo root, with `rsvg-convert` and Python + Pillow:

```bash
cd assets/branding
rsvg-convert -w 24 mark.svg -o /tmp/m24.png
# composite on #121212 or #fff as needed; see prior scripts in git history
```
