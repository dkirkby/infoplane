/** The epoch scrubber: a range input over checkpoint *indices*, plus playback.
 *
 * The slider steps over indices rather than epoch numbers because the checkpoints
 * are log-spaced - one slider step is then one measurement, and the dense early
 * epochs where fitting happens get as much travel as the sparse late ones.
 */
export function scrubber({ slider, readout, playButton, epochs, onChange }) {
  let idx = 0;
  let timer = null;

  slider.min = "0";
  slider.max = String(epochs.length - 1);
  slider.step = "1";
  slider.value = "0";
  slider.setAttribute("aria-label", "Training epoch checkpoint");

  const paint = () => {
    const e = epochs[idx];
    if (readout) readout.textContent = e === 0 ? "epoch 0 (init)" : `epoch ${e.toLocaleString("en-US")}`;
    slider.setAttribute("aria-valuetext", e === 0 ? "epoch 0, initialisation" : `epoch ${e}`);
  };

  const set = (next, notify = true) => {
    idx = Math.max(0, Math.min(epochs.length - 1, next));
    slider.value = String(idx);
    paint();
    if (notify) onChange(idx);
  };

  const stop = () => {
    if (timer) clearInterval(timer);
    timer = null;
    if (playButton) {
      playButton.setAttribute("aria-pressed", "false");
      playButton.textContent = "Play";
    }
  };

  const play = () => {
    if (timer) return stop();
    if (idx >= epochs.length - 1) set(0);
    playButton.setAttribute("aria-pressed", "true");
    playButton.textContent = "Pause";
    timer = setInterval(() => {
      if (idx >= epochs.length - 1) return stop();
      set(idx + 1);
    }, 90);
  };

  slider.addEventListener("input", () => set(Number(slider.value)));
  if (playButton) {
    // Respect reduced-motion by not offering autoplay as the primary path; the
    // slider itself (and its arrow keys) remains the full-fidelity control.
    playButton.addEventListener("click", play);
    if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) playButton.hidden = true;
  }

  paint();
  return {
    get index() {
      return idx;
    },
    set,
    stop,
  };
}
