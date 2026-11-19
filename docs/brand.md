# VORTEX Money brand guide

**Promise:** *Plan for the storm.* Most money apps sell a smooth 12%. VORTEX Money shows the real range of futures,
crashes included, and coaches people to stay invested through them.

**Voice:** calm, honest, plain. Short sentences, rupees not jargon, ranges not promises. Never "guaranteed", never
"buy/sell". Reward good habits (showing up, staying invested, diversifying), never trading.

**Mark:** three arcs spiralling into a calm centre (`app/public/money/brand/mark.svg`): the storm around you, the plan at
the centre. Always on lime (`#c6f432`) with ink arcs (`#0a0d02`), corner radius 28% of the size. Minimum 20 px.
App icons: `icon-32/180/192/512.png`; social card: `og.png` (1200x630).

**Wordmark:** "VORTEX" bold + "Money" regular, Inter, tracking -0.02em (`Wordmark` in `app/src/money/brand.tsx`).

**Colour:** near-black neutrals (`#08090b` page, `#0f1013` panel, `#15171b` raised), one accent lime `#c6f432` for brand
and primary actions only. Status: good `#8fd14f`, warning `#f0b429`, bad `#ff6b6b`. Chart series (validated for colour
blindness on the dark surface): lime `#74a214`, indigo `#7480e8`, amber `#c47c0c`, coral `#e2475a`; amber and coral are
never shown without direct labels.

**Type:** Inter 400/500/600/700 with tabular numbers. Big numbers 650 weight, -0.045em tracking.

**Motion:** purposeful and quick (180-450 ms): page fade-up with a slight blur, springy presses, numbers count up,
charts draw in, toasts for rewards, a level-up moment with confetti. All motion is off under reduced-motion settings.
