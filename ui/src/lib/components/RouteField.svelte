<script lang="ts">
  // Decorative component with no public props. The typed script keeps Svelte's
  // generated component declaration explicit under strict TypeScript checking.
</script>

<svg class="route-field" aria-hidden="true" viewBox="0 0 1600 900" preserveAspectRatio="xMidYMid slice">
  <defs>
    <linearGradient id="svelte-route-gradient" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="currentColor" stop-opacity="0.04" />
      <stop offset="0.5" stop-color="currentColor" stop-opacity="0.2" />
      <stop offset="1" stop-color="currentColor" stop-opacity="0.02" />
    </linearGradient>
  </defs>
  <path class="route-field__path route-field__path--one" d="M-80 180 C180 50 320 310 560 188 S930 20 1120 220 1450 420 1690 230" />
  <path class="route-field__path route-field__path--two" d="M-120 690 C120 520 340 790 570 610 S920 430 1110 620 1460 820 1690 620" />
  <path class="route-field__path route-field__path--three" d="M260 -40 C420 180 250 380 490 510 S900 650 1040 420 1330 160 1580 330" />
  <g class="route-field__nodes">
    <circle cx="560" cy="188" r="8" />
    <circle cx="1120" cy="220" r="8" />
    <circle cx="570" cy="610" r="8" />
    <circle cx="1110" cy="620" r="8" />
    <circle cx="1040" cy="420" r="8" />
  </g>
</svg>
