# FirstLight Design System

**Version:** 1.0.0
**Last Updated:** 2026-01-26
**Status:** Production Ready

---

## Overview

This design system establishes visual standards for FirstLight's emergency intelligence reporting outputs. The system prioritizes accessibility, clarity, and actionability for diverse audiences including emergency managers, local officials, affected residents, and media.

**Design Principles:**

1. **Clarity over complexity** - Information hierarchy that guides the eye
2. **Accessibility first** - WCAG 2.1 AA compliant, colorblind-safe palettes
3. **Actionable intelligence** - Every element serves decision-making
4. **Print and digital parity** - Works in PDF, web, and field conditions
5. **Calm urgency** - Professional authority without panic-inducing design

---

## 1. Color System

### 1.1 Brand Colors

FirstLight's brand palette conveys reliability, authority, and clarity.

| Name | Hex | RGB | Usage |
|------|-----|-----|-------|
| **FirstLight Navy** | `#1A365D` | `26, 54, 93` | Primary brand, headers, navigation |
| **FirstLight Blue** | `#2C5282` | `44, 82, 130` | Secondary brand, accents |
| **FirstLight Sky** | `#3182CE` | `49, 130, 206` | Interactive elements, links |
| **FirstLight Slate** | `#2D3748` | `45, 55, 72` | Body text, high-contrast |

**Contrast Ratios (on white #FFFFFF):**
- FirstLight Navy: 10.3:1 (AAA)
- FirstLight Blue: 6.2:1 (AA Large, AAA)
- FirstLight Sky: 4.5:1 (AA)
- FirstLight Slate: 11.2:1 (AAA)

### 1.2 Semantic Colors

Status and feedback colors with accessible alternatives.

| Status | Primary Hex | RGB | Accessible Dark | Usage |
|--------|-------------|-----|-----------------|-------|
| **Success** | `#38A169` | `56, 161, 105` | `#276749` | Pass, confirmed, safe |
| **Warning** | `#D69E2E` | `214, 158, 46` | `#975A16` | Caution, review needed |
| **Danger** | `#E53E3E` | `229, 62, 62` | `#C53030` | Fail, hazard, urgent |
| **Info** | `#3182CE` | `49, 130, 206` | `#2B6CB0` | Informational, neutral |

**Contrast Ratios (on white #FFFFFF):**
- Success Green: 3.5:1 (use with large text or dark variant)
- Success Dark: 5.8:1 (AA)
- Warning Amber: 2.4:1 (use dark variant for text)
- Warning Dark: 5.1:1 (AA)
- Danger Red: 4.0:1 (use dark variant for small text)
- Info Blue: 4.5:1 (AA)

### 1.3 Flood Severity Palette (Colorblind Accessible)

This palette has been tested for deuteranopia, protanopia, and tritanopia using the Viridis-inspired progression that maintains perceptual uniformity.

| Severity Level | Name | Hex | RGB | Depth Range | CB-Safe |
|----------------|------|-----|-----|-------------|---------|
| **Level 0** | No Flooding | `#F7FAFC` | `247, 250, 252` | 0 m | Yes |
| **Level 1** | Minor | `#90CDF4` | `144, 205, 244` | 0-0.3 m | Yes |
| **Level 2** | Moderate | `#4299E1` | `66, 153, 225` | 0.3-1.0 m | Yes |
| **Level 3** | Significant | `#2B6CB0` | `43, 108, 176` | 1.0-2.0 m | Yes |
| **Level 4** | Severe | `#2C5282` | `44, 82, 130` | 2.0-3.0 m | Yes |
| **Level 5** | Extreme | `#1A365D` | `26, 54, 93` | >3.0 m | Yes |

**Alternative High-Contrast Palette (for print/low-light):**

| Severity | Hex | RGB | Pattern Overlay |
|----------|-----|-----|-----------------|
| Minor | `#BEE3F8` | `190, 227, 248` | None |
| Moderate | `#63B3ED` | `99, 179, 237` | Light dots |
| Significant | `#3182CE` | `49, 130, 206` | Medium dots |
| Severe | `#2C5282` | `44, 82, 130` | Dense dots |
| Extreme | `#1A365D` | `26, 54, 93` | Solid |

**Why Blue Scale:** Water is universally associated with blue. Using a single-hue progression ensures colorblind users can distinguish severity through luminance differences rather than hue differences.

### 1.4 Wildfire Severity Palette (Colorblind Accessible)

Heat-inspired palette using warm tones with sufficient luminance contrast.

| Severity Level | Name | Hex | RGB | Burn Level | CB-Safe |
|----------------|------|-----|-----|------------|---------|
| **Level 0** | Unburned | `#F7FAFC` | `247, 250, 252` | No burn | Yes |
| **Level 1** | Low | `#FED7AA` | `254, 215, 170` | Surface scorch | Yes |
| **Level 2** | Moderate-Low | `#FDBA74` | `253, 186, 116` | Partial understory | Yes |
| **Level 3** | Moderate | `#F97316` | `249, 115, 22` | Full understory | Yes |
| **Level 4** | Moderate-High | `#EA580C` | `234, 88, 12` | Partial canopy | Yes |
| **Level 5** | High | `#9A3412` | `154, 52, 18` | Full canopy loss | Yes |

**Pattern Overlays for Print:**
- Low severity: Vertical lines
- Moderate: Diagonal lines
- High: Cross-hatch

### 1.5 Confidence/Uncertainty Palette

Shows data reliability without implying severity.

| Level | Name | Hex | RGB | Confidence Range | Visual Treatment |
|-------|------|-----|-----|------------------|------------------|
| **High** | Confirmed | `#276749` | `39, 103, 73` | >85% | Solid fill |
| **Medium** | Likely | `#4A5568` | `74, 85, 104` | 60-85% | 75% opacity |
| **Low** | Possible | `#718096` | `113, 128, 150` | 40-60% | 50% opacity + hash |
| **Very Low** | Uncertain | `#A0AEC0` | `160, 174, 192` | <40% | 25% opacity + dots |

**Important:** Confidence colors are intentionally neutral (gray-green) to avoid confusion with severity colors.

### 1.6 Base Map Palette

Muted colors for context layers that don't compete with data overlays.

| Layer | Hex | RGB | Opacity | Usage |
|-------|-----|-----|---------|-------|
| **Land** | `#EDF2F7` | `237, 242, 247` | 100% | Base landmass |
| **Water Bodies** | `#E2E8F0` | `226, 232, 240` | 100% | Pre-existing water |
| **Urban Areas** | `#CBD5E0` | `203, 213, 224` | 80% | Built-up areas |
| **Roads Major** | `#A0AEC0` | `160, 174, 192` | 100% | Highways, arterials |
| **Roads Minor** | `#CBD5E0` | `203, 213, 224` | 70% | Local streets |
| **Boundaries** | `#718096` | `113, 128, 150` | 100% | Admin boundaries |
| **Labels** | `#4A5568` | `74, 85, 104` | 100% | Place names |

### 1.7 Neutral Scale

For UI elements, backgrounds, and typography.

| Token | Hex | RGB | Usage |
|-------|-----|-----|-------|
| `neutral-50` | `#F7FAFC` | `247, 250, 252` | Lightest backgrounds |
| `neutral-100` | `#EDF2F7` | `237, 242, 247` | Card backgrounds |
| `neutral-200` | `#E2E8F0` | `226, 232, 240` | Borders, dividers |
| `neutral-300` | `#CBD5E0` | `203, 213, 224` | Disabled states |
| `neutral-400` | `#A0AEC0` | `160, 174, 192` | Placeholder text |
| `neutral-500` | `#718096` | `113, 128, 150` | Secondary text |
| `neutral-600` | `#4A5568` | `74, 85, 104` | Body text |
| `neutral-700` | `#2D3748` | `45, 55, 72` | Headings |
| `neutral-800` | `#1A202C` | `26, 32, 44` | High emphasis |
| `neutral-900` | `#171923` | `23, 25, 35` | Maximum contrast |

---

## 2. Typography System

### 2.1 Font Stack

**Primary Font (Headlines and UI):**
```css
font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
             'Helvetica Neue', Arial, sans-serif;
```

**Secondary Font (Body Text):**
```css
font-family: 'Source Sans Pro', -apple-system, BlinkMacSystemFont,
             'Segoe UI', Roboto, sans-serif;
```

**Monospace (Data, Coordinates):**
```css
font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono',
             'Source Code Pro', monospace;
```

**Rationale:** System fonts ensure fast loading, consistent rendering across platforms, and high readability without external dependencies. Source Sans Pro is specified for body text when available as it has excellent legibility at small sizes.

### 2.2 Type Scale

Base size: 16px (1rem)
Scale ratio: 1.25 (Major Third)

| Token | Size | Weight | Line Height | Letter Spacing | Usage |
|-------|------|--------|-------------|----------------|-------|
| `text-xs` | 12px / 0.75rem | 400 | 1.5 | 0.02em | Captions, timestamps, metadata |
| `text-sm` | 14px / 0.875rem | 400 | 1.5 | 0.01em | Secondary text, table data |
| `text-base` | 16px / 1rem | 400 | 1.6 | 0 | Body text, paragraphs |
| `text-lg` | 18px / 1.125rem | 500 | 1.5 | -0.01em | Lead paragraphs, emphasis |
| `text-xl` | 20px / 1.25rem | 600 | 1.4 | -0.01em | H4, card titles, section headers |
| `text-2xl` | 24px / 1.5rem | 600 | 1.3 | -0.02em | H3, major sections |
| `text-3xl` | 30px / 1.875rem | 700 | 1.2 | -0.02em | H2, page sections |
| `text-4xl` | 36px / 2.25rem | 700 | 1.1 | -0.02em | H1, report titles |
| `text-5xl` | 48px / 3rem | 800 | 1.1 | -0.03em | Hero headlines, cover pages |

### 2.3 Map Typography

Special considerations for cartographic text.

| Element | Size | Weight | Color | Halo | Usage |
|---------|------|--------|-------|------|-------|
| **Major City** | 14px | 700 | `#1A202C` | 2px white | Primary place names |
| **Town/Area** | 12px | 600 | `#2D3748` | 1.5px white | Secondary places |
| **Neighborhood** | 11px | 500 | `#4A5568` | 1px white | Local areas |
| **Road Label** | 10px | 500 | `#4A5568` | 1px white | Street names |
| **Water Label** | 11px | 400 italic | `#2B6CB0` | 1px white | Rivers, lakes |
| **Legend Title** | 12px | 700 | `#1A202C` | None | Legend headers |
| **Legend Item** | 11px | 400 | `#2D3748` | None | Legend entries |
| **Scale Bar** | 10px | 500 | `#4A5568` | None | Distance markers |

**Halo Effect CSS:**
```css
text-shadow:
  -1px -1px 0 white,
  1px -1px 0 white,
  -1px 1px 0 white,
  1px 1px 0 white;
```

### 2.4 Typography Hierarchy Rules

**Reports:**
- H1: Report title only (one per document)
- H2: Major sections (Executive Summary, Analysis, Recommendations)
- H3: Subsections within major sections
- H4: Component headers (cards, tables)
- Body: All narrative content
- Small: Metadata, sources, timestamps

**Data Presentation:**
- Large numbers: `text-2xl` to `text-4xl`, bold
- Units/labels: `text-sm`, regular weight, muted color
- Table headers: `text-sm`, semi-bold, uppercase
- Table data: `text-sm`, regular

**Line Length:**
- Optimal: 65-75 characters per line
- Maximum: 90 characters
- Minimum: 45 characters

---

## 3. Iconography

### 3.1 Emergency/Disaster Icons

Standard icons for event types. Use outlined style for UI, filled for emphasis.

| Icon | Name | Unicode/Source | Usage |
|------|------|----------------|-------|
| Flood/Water | `water-drop` | Heroicons | Flood events, water levels |
| Fire | `fire` | Heroicons | Wildfire, burn areas |
| Storm/Wind | `cloud-bolt` | Heroicons (custom) | Hurricanes, storms |
| Warning | `exclamation-triangle` | Heroicons | Alerts, hazards |
| Danger | `x-circle` | Heroicons | Critical hazard |
| Safe | `check-circle` | Heroicons | Cleared areas |
| Information | `information-circle` | Heroicons | General info |

**Recommended Icon Library:** [Heroicons](https://heroicons.com/) - MIT licensed, consistent style, available in outline and solid variants.

### 3.2 Infrastructure Icons

| Icon | Name | Usage | Notes |
|------|------|-------|-------|
| Hospital | `building-hospital` | Medical facilities | Cross symbol |
| School | `academic-cap` | Schools, universities | |
| Shelter | `home-modern` | Emergency shelters | With roof emphasis |
| Fire Station | `fire-truck` | Fire departments | |
| Police | `shield-check` | Law enforcement | |
| Government | `building-office` | Government buildings | |
| Power | `bolt` | Electrical infrastructure | |
| Water Treatment | `beaker` | Water/sewage facilities | |
| Bridge | `bridge` | Transportation infrastructure | Custom |
| Airport | `airplane` | Airports, helipads | |

### 3.3 Status Indicators

| Status | Icon | Color | Animation |
|--------|------|-------|-----------|
| Loading | `spinner` | `neutral-500` | Rotate |
| Success | `check` | `success` | None |
| Warning | `exclamation` | `warning` | None |
| Error | `x-mark` | `danger` | None |
| Info | `information` | `info` | None |
| New/Updated | `dot` | `info` | Pulse |

### 3.4 Map Symbols

**Point Symbols:**
```
Size: 24px default, scale with zoom
Stroke: 2px white for contrast
Shadow: 0 1px 3px rgba(0,0,0,0.3)
```

| Feature | Shape | Fill | Stroke |
|---------|-------|------|--------|
| Hospital | Circle + Cross | `#E53E3E` | White |
| School | Circle + Cap | `#3182CE` | White |
| Shelter | Triangle + House | `#38A169` | White |
| Fire Station | Diamond | `#F97316` | White |
| Affected Location | Pin | `#E53E3E` | White |
| Safe Location | Pin | `#38A169` | White |

---

## 4. Component Library

### 4.1 Metric Cards

Display key statistics with visual hierarchy.

```
+------------------------------------------+
|  3,026                                   |
|  Hectares Flooded                        |
|  [progress bar ███████░░░] 28.9%        |
+------------------------------------------+
```

**Specifications:**

| Property | Value |
|----------|-------|
| Background | `#FFFFFF` |
| Border | `1px solid neutral-200` |
| Border Radius | `8px` |
| Padding | `24px` |
| Shadow | `0 1px 3px rgba(0,0,0,0.1)` |
| Number Size | `text-4xl` (36px) |
| Number Weight | `700` |
| Number Color | `neutral-700` |
| Label Size | `text-sm` (14px) |
| Label Color | `neutral-500` |

**Status Variants:**
- Success: Left border `4px solid #38A169`
- Warning: Left border `4px solid #D69E2E`
- Danger: Left border `4px solid #E53E3E`

### 4.2 Alert/Warning Boxes

```
+------------------------------------------+
| [!] FLOOD WARNING                        |
|                                          |
| Major flooding detected in the Fort      |
| Myers area. Avoid travel through         |
| affected zones.                          |
|                                          |
| Posted: Jan 26, 2026 at 3:45 PM EST     |
+------------------------------------------+
```

**Alert Levels:**

| Level | Background | Border | Icon Color | Text |
|-------|------------|--------|------------|------|
| Critical | `#FED7D7` | `4px #E53E3E` | `#C53030` | `#742A2A` |
| Warning | `#FEFCBF` | `4px #D69E2E` | `#975A16` | `#744210` |
| Info | `#BEE3F8` | `4px #3182CE` | `#2B6CB0` | `#2A4365` |
| Success | `#C6F6D5` | `4px #38A169` | `#276749` | `#22543D` |

**Specifications:**

| Property | Value |
|----------|-------|
| Border Radius | `8px` |
| Padding | `16px 20px` |
| Icon Size | `24px` |
| Title Size | `text-lg` (18px), weight 600 |
| Body Size | `text-base` (16px) |
| Timestamp Size | `text-xs` (12px) |

### 4.3 Data Tables

```
+--------+----------+--------+-----------+
| Region | Hectares | Status | Conf.     |
+--------+----------+--------+-----------+
| Zone A |    1,245 | Severe | 92%       |
| Zone B |      892 | Mod.   | 85%       |
| Zone C |      421 | Minor  | 78%       |
+--------+----------+--------+-----------+
```

**Specifications:**

| Property | Value |
|----------|-------|
| Header Background | `neutral-100` |
| Header Text | `text-xs`, uppercase, weight 600, `neutral-600` |
| Header Padding | `12px 16px` |
| Row Background | Alternating `#FFFFFF` / `neutral-50` |
| Row Padding | `12px 16px` |
| Cell Text | `text-sm`, `neutral-700` |
| Number Alignment | Right |
| Text Alignment | Left |
| Border | `1px solid neutral-200` |

**Interactive States (web):**
- Hover: Background `neutral-100`
- Selected: Background `#EBF8FF`, left border `3px #3182CE`

### 4.4 Legend Designs

**Continuous Scale Legend (for flood depth, etc.):**

```
Flood Depth (meters)
[gradient bar ░░▒▒▓▓██]
0.0       1.5       3.0+
```

| Property | Value |
|----------|-------|
| Title Size | `text-sm`, weight 600 |
| Bar Height | `16px` |
| Bar Width | `200px` (min), `300px` (max) |
| Label Size | `text-xs` |
| Border Radius | `4px` |

**Categorical Legend:**

```
Flood Severity
● Minor (0-0.3m)
● Moderate (0.3-1m)
● Significant (1-2m)
● Severe (2-3m)
● Extreme (3m+)
```

| Property | Value |
|----------|-------|
| Title Size | `text-sm`, weight 600 |
| Symbol Size | `12px` diameter |
| Symbol Shape | Circle for areas, line for linear features |
| Label Size | `text-xs` |
| Row Gap | `8px` |

### 4.5 Before/After Comparison Layout

**Side-by-Side Layout:**

```
+---------------------+  +---------------------+
|                     |  |                     |
|   BEFORE            |  |   AFTER             |
|   Sep 25, 2022      |  |   Sep 29, 2022      |
|                     |  |                     |
|   [satellite        |  |   [satellite        |
|    image]           |  |    image with       |
|                     |  |    flood overlay]   |
|                     |  |                     |
+---------------------+  +---------------------+
```

| Property | Value |
|----------|-------|
| Container Gap | `16px` |
| Image Aspect Ratio | 4:3 or 16:9 (consistent) |
| Label Position | Top-left overlay |
| Label Background | `rgba(0,0,0,0.7)` |
| Label Text | `text-sm`, white, weight 600 |
| Label Padding | `8px 12px` |
| Border Radius | `4px` |

**Slider Layout (interactive):**

```
+----------------------------------------+
|                                        |
|   ◄─────────[SLIDER]─────────►        |
|                                        |
|   Before          |         After      |
|   Sep 25, 2022    |         Sep 29     |
|                                        |
+----------------------------------------+
```

### 4.6 Severity Badges

```
[SEVERE]  [MODERATE]  [MINOR]  [UNCERTAIN]
```

| Severity | Background | Text | Border |
|----------|------------|------|--------|
| Extreme | `#1A365D` | White | None |
| Severe | `#2C5282` | White | None |
| Significant | `#3182CE` | White | None |
| Moderate | `#90CDF4` | `#1A365D` | None |
| Minor | `#E2E8F0` | `#2D3748` | None |
| Uncertain | `#FFFFFF` | `#718096` | `1px #CBD5E0` |

**Specifications:**

| Property | Value |
|----------|-------|
| Padding | `4px 12px` |
| Border Radius | `4px` |
| Font Size | `text-xs` (12px) |
| Font Weight | `600` |
| Text Transform | Uppercase |
| Letter Spacing | `0.05em` |

---

## 5. Layout System

### 5.1 Grid System

**12-Column Grid:**

| Breakpoint | Name | Width | Columns | Gutter |
|------------|------|-------|---------|--------|
| `xs` | Mobile | <576px | 4 | 16px |
| `sm` | Mobile Landscape | 576-767px | 6 | 16px |
| `md` | Tablet | 768-991px | 8 | 24px |
| `lg` | Desktop | 992-1199px | 12 | 24px |
| `xl` | Large Desktop | 1200-1399px | 12 | 32px |
| `xxl` | Extra Large | >=1400px | 12 | 32px |

### 5.2 Spacing Scale

Base: 4px

| Token | Value | Usage |
|-------|-------|-------|
| `space-1` | 4px | Tight gaps, icon spacing |
| `space-2` | 8px | Related elements |
| `space-3` | 12px | Form elements |
| `space-4` | 16px | Card padding, standard gaps |
| `space-5` | 20px | Section padding (small) |
| `space-6` | 24px | Component separation |
| `space-8` | 32px | Section gaps |
| `space-10` | 40px | Major section breaks |
| `space-12` | 48px | Page section padding |
| `space-16` | 64px | Hero padding |
| `space-20` | 80px | Section separators |
| `space-24` | 96px | Major page divisions |

### 5.3 Report Page Layout

**Executive Summary (1-page):**
```
+--------------------------------------------------+
|  [LOGO]                    [DATE/TIME]           |  Header: 60px
+--------------------------------------------------+
|                                                  |
|  HURRICANE IAN FLOOD ANALYSIS                    |  Title: 80px
|  Fort Myers, FL | Category 4                     |
|                                                  |
+--------------------------------------------------+
|  +--------+ +--------+ +--------+ +--------+     |
|  | 3,026  | | 28.9%  | |  HIGH  | | 6/8    |     |  Stats: 100px
|  | ha     | | cover  | | conf.  | | QC     |     |
|  +--------+ +--------+ +--------+ +--------+     |
+--------------------------------------------------+
|                    |                             |
|   [MINI MAP]       |   What Happened             |  Main: 400px
|   showing flood    |   Plain language summary    |
|   extent           |   of the flood event and    |
|                    |   its impacts.              |
|                    |                             |
+--------------------------------------------------+
|  Who Is Affected              | Emergency Info   |  Footer: 120px
|  - Est. population            | Phone: XXX       |
|  - Critical facilities        | Web: XXX         |
+--------------------------------------------------+
```

**Content Margins:**
- Print: 0.75" all sides (54px at 72dpi)
- Screen: 32px horizontal, 24px vertical

---

## 6. Accessibility Guidelines

### 6.1 WCAG 2.1 AA Requirements

**Color Contrast:**
- Normal text (< 18px): Minimum 4.5:1 contrast ratio
- Large text (>= 18px bold or >= 24px): Minimum 3:1 contrast ratio
- UI components and graphics: Minimum 3:1 contrast ratio

**Color Independence:**
- Never use color as the only means of conveying information
- Always pair with text labels, patterns, or icons
- Severity levels must include text labels, not just colors

**Focus States:**
```css
:focus {
  outline: 2px solid #3182CE;
  outline-offset: 2px;
}
:focus:not(:focus-visible) {
  outline: none;
}
:focus-visible {
  outline: 2px solid #3182CE;
  outline-offset: 2px;
}
```

### 6.2 Colorblind Accessibility

**Testing Protocol:**
1. Simulate deuteranopia (red-green, most common)
2. Simulate protanopia (red-green)
3. Simulate tritanopia (blue-yellow, rare)

**Tools:**
- Sim Daltonism (Mac)
- Color Oracle (Cross-platform)
- Stark (Figma plugin)

**Flood Palette Verification:**

| Original | Deuteranopia | Protanopia | Tritanopia | Distinguishable |
|----------|--------------|------------|------------|-----------------|
| `#90CDF4` | `#9AC5E8` | `#9FCBE5` | `#85D4F1` | Yes |
| `#4299E1` | `#5A97D5` | `#6199D0` | `#4BA0DD` | Yes |
| `#2B6CB0` | `#3D6AAA` | `#456BA3` | `#3472AB` | Yes |
| `#2C5282` | `#3A5180` | `#42527B` | `#35577D` | Yes |
| `#1A365D` | `#28365C` | `#2F365A` | `#223958` | Yes |

### 6.3 Screen Reader Considerations

**Images:**
- All images must have descriptive `alt` text
- Decorative images: `alt=""`
- Maps: Provide text summary of key findings

**Example alt text for flood map:**
```html
<img alt="Map showing flood extent in Fort Myers, Florida following
Hurricane Ian. Approximately 3,026 hectares are flooded, primarily
concentrated in downtown Fort Myers and the Caloosahatchee River
corridor. Severity ranges from minor (light blue) to severe (dark blue)."
```

**Data Tables:**
- Use proper `<th>` headers with `scope` attributes
- Include `<caption>` for table context
- Avoid merged cells when possible

**Interactive Elements:**
- All interactive elements must be keyboard accessible
- Announce state changes with ARIA live regions
- Provide skip links for repeated content

### 6.4 Print-Friendly Requirements

**Color Adjustments:**
- Ensure sufficient contrast at 100% black ink
- Provide pattern overlays for severity levels
- Use darker variants of semantic colors

**Typography:**
- Minimum body text: 10pt (13.3px)
- Minimum caption text: 8pt (10.7px)
- Line height minimum: 1.4

**Layout:**
- Avoid elements that split across page breaks
- Provide page numbers
- Include "Continued on next page" indicators for long tables

**Print Stylesheet:**
```css
@media print {
  .no-print { display: none; }
  body { font-size: 10pt; }
  a[href]:after { content: " (" attr(href) ")"; }
  .page-break { page-break-before: always; }
}
```

---

## 7. Motion and Animation

### 7.1 Timing Functions

| Type | Duration | Easing | Usage |
|------|----------|--------|-------|
| Instant | 0ms | - | Immediate feedback |
| Fast | 100ms | `ease-out` | Micro-interactions |
| Normal | 200ms | `ease-in-out` | Most transitions |
| Slow | 300ms | `ease-in-out` | Page transitions |
| Emphasis | 400ms | `cubic-bezier(0.4, 0, 0.2, 1)` | Attention |

### 7.2 Animation Guidelines

**Do animate:**
- Hover/focus states
- Dropdown/modal appearances
- Loading indicators
- Progress bars
- Before/after slider position

**Do not animate:**
- Critical emergency information
- Color changes on static data
- Layout shifts after load
- Anything that could cause motion sickness

**Reduced Motion:**
```css
@media (prefers-reduced-motion: reduce) {
  * {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
```

---

## 8. Design Tokens (CSS Custom Properties)

```css
:root {
  /* Brand Colors */
  --color-brand-navy: #1A365D;
  --color-brand-blue: #2C5282;
  --color-brand-sky: #3182CE;
  --color-brand-slate: #2D3748;

  /* Semantic Colors */
  --color-success: #38A169;
  --color-success-dark: #276749;
  --color-warning: #D69E2E;
  --color-warning-dark: #975A16;
  --color-danger: #E53E3E;
  --color-danger-dark: #C53030;
  --color-info: #3182CE;
  --color-info-dark: #2B6CB0;

  /* Flood Severity */
  --color-flood-none: #F7FAFC;
  --color-flood-minor: #90CDF4;
  --color-flood-moderate: #4299E1;
  --color-flood-significant: #2B6CB0;
  --color-flood-severe: #2C5282;
  --color-flood-extreme: #1A365D;

  /* Wildfire Severity */
  --color-fire-none: #F7FAFC;
  --color-fire-low: #FED7AA;
  --color-fire-mod-low: #FDBA74;
  --color-fire-moderate: #F97316;
  --color-fire-mod-high: #EA580C;
  --color-fire-high: #9A3412;

  /* Confidence */
  --color-conf-high: #276749;
  --color-conf-medium: #4A5568;
  --color-conf-low: #718096;
  --color-conf-very-low: #A0AEC0;

  /* Neutrals */
  --color-neutral-50: #F7FAFC;
  --color-neutral-100: #EDF2F7;
  --color-neutral-200: #E2E8F0;
  --color-neutral-300: #CBD5E0;
  --color-neutral-400: #A0AEC0;
  --color-neutral-500: #718096;
  --color-neutral-600: #4A5568;
  --color-neutral-700: #2D3748;
  --color-neutral-800: #1A202C;
  --color-neutral-900: #171923;

  /* Typography */
  --font-sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
               'Helvetica Neue', Arial, sans-serif;
  --font-mono: 'SF Mono', Monaco, Inconsolata, 'Roboto Mono',
               'Source Code Pro', monospace;

  /* Font Sizes */
  --text-xs: 0.75rem;
  --text-sm: 0.875rem;
  --text-base: 1rem;
  --text-lg: 1.125rem;
  --text-xl: 1.25rem;
  --text-2xl: 1.5rem;
  --text-3xl: 1.875rem;
  --text-4xl: 2.25rem;
  --text-5xl: 3rem;

  /* Spacing */
  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --space-5: 20px;
  --space-6: 24px;
  --space-8: 32px;
  --space-10: 40px;
  --space-12: 48px;
  --space-16: 64px;
  --space-20: 80px;
  --space-24: 96px;

  /* Border Radius */
  --radius-sm: 4px;
  --radius-md: 8px;
  --radius-lg: 12px;
  --radius-xl: 16px;
  --radius-full: 9999px;

  /* Shadows */
  --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
  --shadow-md: 0 4px 6px rgba(0,0,0,0.07);
  --shadow-lg: 0 10px 15px rgba(0,0,0,0.1);
  --shadow-xl: 0 20px 25px rgba(0,0,0,0.15);

  /* Transitions */
  --transition-fast: 100ms ease-out;
  --transition-normal: 200ms ease-in-out;
  --transition-slow: 300ms ease-in-out;
}
```

---

## 9. Usage Examples

### 9.1 Severity Badge Implementation

```html
<span class="severity-badge severity-severe">SEVERE</span>

<style>
.severity-badge {
  display: inline-block;
  padding: 4px 12px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.severity-extreme { background: #1A365D; color: white; }
.severity-severe { background: #2C5282; color: white; }
.severity-significant { background: #3182CE; color: white; }
.severity-moderate { background: #90CDF4; color: #1A365D; }
.severity-minor { background: #E2E8F0; color: #2D3748; }
.severity-uncertain {
  background: white;
  color: #718096;
  border: 1px solid #CBD5E0;
}
</style>
```

### 9.2 Alert Box Implementation

```html
<div class="alert alert-warning">
  <div class="alert-icon">
    <svg><!-- warning icon --></svg>
  </div>
  <div class="alert-content">
    <h4 class="alert-title">FLOOD WARNING</h4>
    <p class="alert-body">Major flooding detected in the Fort Myers area.</p>
    <span class="alert-timestamp">Posted: Jan 26, 2026 at 3:45 PM EST</span>
  </div>
</div>

<style>
.alert {
  display: flex;
  gap: 16px;
  padding: 16px 20px;
  border-radius: 8px;
  border-left: 4px solid;
}

.alert-warning {
  background: #FEFCBF;
  border-color: #D69E2E;
}

.alert-warning .alert-icon { color: #975A16; }
.alert-warning .alert-title { color: #744210; }
.alert-warning .alert-body { color: #744210; }

.alert-icon { flex-shrink: 0; width: 24px; height: 24px; }
.alert-title { font-size: 18px; font-weight: 600; margin: 0 0 8px; }
.alert-body { font-size: 16px; margin: 0 0 8px; }
.alert-timestamp { font-size: 12px; color: #718096; }
</style>
```

### 9.3 Metric Card Implementation

```html
<div class="metric-card metric-card-warning">
  <div class="metric-value">3,026</div>
  <div class="metric-label">Hectares Flooded</div>
  <div class="metric-subtext">28.9% of analysis area</div>
</div>

<style>
.metric-card {
  background: white;
  border: 1px solid #E2E8F0;
  border-radius: 8px;
  padding: 24px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.metric-card-warning { border-left: 4px solid #D69E2E; }
.metric-card-danger { border-left: 4px solid #E53E3E; }
.metric-card-success { border-left: 4px solid #38A169; }

.metric-value {
  font-size: 36px;
  font-weight: 700;
  color: #2D3748;
  line-height: 1.1;
}

.metric-label {
  font-size: 14px;
  color: #718096;
  margin-top: 8px;
}

.metric-subtext {
  font-size: 12px;
  color: #A0AEC0;
  margin-top: 4px;
}
</style>
```

---

## 10. File Format Specifications

### 10.1 Image Exports

| Format | Resolution | Color Space | Use Case |
|--------|------------|-------------|----------|
| PNG | 144 DPI | sRGB | Web, thumbnails |
| PNG | 300 DPI | sRGB | Print reports |
| SVG | Vector | sRGB | Icons, logos |
| GeoTIFF | Native | N/A | Geospatial data |
| WebP | 144 DPI | sRGB | Web (optimized) |

### 10.2 Report Exports

| Format | Dimensions | Margins | Notes |
|--------|------------|---------|-------|
| PDF (US Letter) | 8.5" x 11" | 0.75" | Standard reports |
| PDF (A4) | 210mm x 297mm | 19mm | International |
| HTML | Responsive | 32px | Interactive viewing |
| PNG (Social) | 1200x630px | N/A | OpenGraph/Twitter |

---

## Appendix A: Color Accessibility Testing Results

### Flood Palette - Contrast Ratios

| Foreground | Background | Ratio | AA Normal | AA Large | AAA |
|------------|------------|-------|-----------|----------|-----|
| White | `#1A365D` | 10.3:1 | Pass | Pass | Pass |
| White | `#2C5282` | 6.2:1 | Pass | Pass | Pass |
| White | `#2B6CB0` | 4.8:1 | Pass | Pass | Fail |
| White | `#4299E1` | 3.2:1 | Fail | Pass | Fail |
| `#1A365D` | `#90CDF4` | 5.7:1 | Pass | Pass | Fail |
| `#1A365D` | White | 10.3:1 | Pass | Pass | Pass |

### Semantic Colors - Contrast Ratios

| Color | On White | On Dark (`#171923`) | Recommendation |
|-------|----------|---------------------|----------------|
| Success `#38A169` | 3.5:1 | 5.2:1 | Use dark variant for text |
| Success Dark `#276749` | 5.8:1 | 3.6:1 | Use for text on white |
| Warning `#D69E2E` | 2.4:1 | 6.1:1 | Never use for text alone |
| Warning Dark `#975A16` | 5.1:1 | 4.2:1 | Use for text on white |
| Danger `#E53E3E` | 4.0:1 | 5.0:1 | OK for large text only |
| Danger Dark `#C53030` | 5.4:1 | 4.0:1 | Use for small text |

---

## Appendix B: Glossary of Terms

For non-technical users of this design system:

| Term | Definition |
|------|------------|
| **Contrast Ratio** | Measure of brightness difference between text and background. Higher is more readable. |
| **WCAG** | Web Content Accessibility Guidelines - international standards for web accessibility. |
| **AA/AAA** | Accessibility compliance levels. AA is standard, AAA is enhanced. |
| **Colorblind Safe** | Can be distinguished by users with color vision deficiency. |
| **rem** | Relative unit based on root font size (usually 16px). |
| **Halo** | White outline around text to improve readability on varied backgrounds. |
| **Semantic Color** | Color that conveys meaning (success, warning, danger, info). |

---

*FirstLight Design System v1.0.0 - For questions, contact the design team.*
