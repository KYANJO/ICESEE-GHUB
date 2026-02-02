// _static/icesee-sidebar.js
// Click-to-collapse for sidebar section captions (targets the correct section).
(function () {
  function ready(fn) {
    if (document.readyState !== "loading") fn();
    else document.addEventListener("DOMContentLoaded", fn);
  }

  function findNextULAfterCaption(captionEl) {
    // In sphinx-book-theme, the structure is usually:
    // <p class="caption">...</p>
    // <ul>...</ul>
    //
    // Sometimes there are small wrapper elements, so walk forward until we find a UL.
    let el = captionEl.nextElementSibling;
    while (el && el.tagName !== "UL") {
      el = el.nextElementSibling;
    }
    return el && el.tagName === "UL" ? el : null;
  }

  ready(function () {
    const captions = Array.from(document.querySelectorAll("nav.bd-links p.caption"));

    captions.forEach((cap) => {
      const ul = findNextULAfterCaption(cap);
      if (!ul) return;

      cap.classList.add("icesee-caption");
      cap.setAttribute("role", "button");
      cap.setAttribute("tabindex", "0");

      // If the UL is currently visible, treat as expanded.
      const isCollapsed = ul.classList.contains("icesee-collapsed");
      cap.setAttribute("aria-expanded", isCollapsed ? "false" : "true");

      function toggle() {
        const collapsedNow = ul.classList.toggle("icesee-collapsed");
        cap.setAttribute("aria-expanded", collapsedNow ? "false" : "true");
      }

      cap.addEventListener("click", toggle);
      cap.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          toggle();
        }
      });
    });
  });
})();