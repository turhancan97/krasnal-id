/* Progressive enhancement for the project page; no model or dataset downloads. */
document.title = document.getElementById("paper-title").textContent;

const copyButton = document.getElementById("copy-citation");
const copyStatus = document.getElementById("copy-status");
copyButton.addEventListener("click", async () => {
  const citation = document.getElementById("bibtex").textContent;
  try {
    if (!navigator.clipboard?.writeText) throw new Error("Clipboard API unavailable");
    await navigator.clipboard.writeText(citation);
    copyStatus.textContent = "Citation copied.";
  } catch {
    // Older browsers and denied clipboard permissions can still copy a selection.
    const selection = window.getSelection();
    const savedRanges = Array.from({ length: selection.rangeCount }, (_, i) =>
      selection.getRangeAt(i).cloneRange(),
    );
    const field = document.createElement("textarea");
    field.value = citation;
    field.setAttribute("readonly", "");
    field.setAttribute("aria-label", "Software citation");
    field.style.position = "fixed";
    field.style.opacity = "0";
    document.body.append(field);
    field.select();
    let copied = false;
    try {
      copied = document.execCommand("copy");
    } finally {
      field.remove();
      copyButton.focus({ preventScroll: true });
      selection.removeAllRanges();
      savedRanges.forEach((range) => selection.addRange(range));
    }
    if (copied) {
      copyStatus.textContent = "Citation copied.";
    } else {
      const range = document.createRange();
      range.selectNodeContents(document.getElementById("bibtex"));
      selection.removeAllRanges();
      selection.addRange(range);
      copyStatus.textContent = "Citation selected. Use your browser’s Copy command.";
    }
  }
});
