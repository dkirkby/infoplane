/** Light/dark/system toggle.
 *
 * Dark mode is a selected palette, not an inverted one (see main.css), so all this
 * does is stamp `data-theme` on <html>; the stylesheet owns every colour. The
 * stamp must beat the OS setting in both directions, which is why the stylesheet
 * declares the dark values under both a guarded media query and this attribute.
 */
const KEY = "infoplane-theme";
const ORDER = ["system", "light", "dark"];
const LABEL = { system: "Theme: system", light: "Theme: light", dark: "Theme: dark" };

function read() {
  try {
    const v = localStorage.getItem(KEY);
    return ORDER.includes(v) ? v : "system";
  } catch {
    return "system"; // private windows / blocked storage
  }
}

function apply(mode) {
  if (mode === "system") document.documentElement.removeAttribute("data-theme");
  else document.documentElement.setAttribute("data-theme", mode);
}

export function initTheme(button) {
  let mode = read();
  apply(mode);
  if (!button) return;
  const paint = () => {
    button.textContent = LABEL[mode];
    button.setAttribute("aria-label", LABEL[mode]);
  };
  paint();
  button.addEventListener("click", () => {
    mode = ORDER[(ORDER.indexOf(mode) + 1) % ORDER.length];
    apply(mode);
    try {
      localStorage.setItem(KEY, mode);
    } catch {
      /* non-fatal: the theme just won't persist */
    }
    paint();
  });
}
