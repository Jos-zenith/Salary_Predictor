/**
 * Shared front-end behaviour for CompIntel.
 * Handles the mobile nav toggle and the simulated live-signal feed.
 * No scroll-fade choreography, no decorative motion — content is visible
 * immediately; the only things that animate are real state changes.
 */
document.addEventListener("DOMContentLoaded", () => {
  // Mobile navigation
  const navToggle = document.querySelector(".nav-toggle");
  const navLinks = document.querySelector(".nav-links");
  if (navToggle && navLinks) {
    navToggle.addEventListener("click", () => {
      navLinks.classList.toggle("nav-open");
      navToggle.classList.toggle("is-active");
    });
    navLinks.querySelectorAll("a").forEach((link) => {
      link.addEventListener("click", () => {
        navLinks.classList.remove("nav-open");
        navToggle.classList.remove("is-active");
      });
    });
  }

  // Sidebar toggle (mobile, app-shell pages)
  const sidebarToggle = document.getElementById("sidebar-toggle");
  const sidebar = document.getElementById("app-sidebar");
  if (sidebarToggle && sidebar) {
    sidebarToggle.addEventListener("click", () => sidebar.classList.toggle("is-open"));
    document.addEventListener("click", (event) => {
      if (sidebar.classList.contains("is-open") && !sidebar.contains(event.target) && event.target !== sidebarToggle) {
        sidebar.classList.remove("is-open");
      }
    });
  }

  // Simulated live-signal feed (see market_signals.py — not a real external feed).
  // Shown in exactly one place: the landing-page demo terminal (#terminal-feed).
  // It used to also duplicate into every tool page's topbar, which put fake
  // "hiring surge" chatter on pages like Compliance Check that have nothing to
  // do with market signals — removed for that reason.
  const terminal = document.getElementById("terminal-feed");
  if (terminal && "EventSource" in window) {
    const MAX_ITEMS = 6;
    const source = new EventSource("/live-signals");
    source.onmessage = (event) => {
      let data;
      try {
        data = JSON.parse(event.data);
      } catch (err) {
        return;
      }
      const empty = terminal.querySelector(".t-empty");
      if (empty) empty.remove();
      const div = document.createElement("div");
      div.className = "t-line";
      const ts = document.createElement("span");
      ts.className = "t-ts";
      ts.textContent = data.ts;
      div.appendChild(ts);
      div.appendChild(document.createTextNode(data.message));
      terminal.appendChild(div);
      while (terminal.children.length > MAX_ITEMS) {
        terminal.removeChild(terminal.firstChild);
      }
      terminal.scrollTop = terminal.scrollHeight;
    };
    source.onerror = () => {
      // Dev-server reload or connection drop — browser will retry automatically.
    };
  }
});
