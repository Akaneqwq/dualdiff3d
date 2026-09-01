(() => {
  "use strict";

  const root = document.documentElement;
  const nav = document.querySelector(".site-nav");
  const themeToggle = document.querySelector("#theme-toggle");
  const themeMeta = document.querySelector('meta[name="theme-color"]');
  const menuToggle = document.querySelector("#menu-toggle");
  const mobileMenu = document.querySelector("#mobile-menu");
  const videos = [...document.querySelectorAll(".video-grid video")];
  const playControl = document.querySelector("#play-control");
  const sceneButtons = [...document.querySelectorAll("[data-scene]")];
  let playing = false;
  let resultVisible = false;
  let sceneLoadId = 0;

  const setTheme = (theme) => {
    root.dataset.theme = theme;
    localStorage.setItem("dualdiff3d-theme", theme);
    themeMeta?.setAttribute("content", theme === "dark" ? "#0f172a" : "#fefffe");
  };

  themeToggle?.addEventListener("click", () => {
    setTheme(root.dataset.theme === "dark" ? "light" : "dark");
  });
  setTheme(root.dataset.theme || "light");

  const arxivLink = document.querySelector("#arxiv-link");
  if (arxivLink?.getAttribute("href") === "#") {
    arxivLink.addEventListener("click", (event) => event.preventDefault());
  } else {
    arxivLink?.removeAttribute("aria-disabled");
    arxivLink?.setAttribute("target", "_blank");
    arxivLink?.setAttribute("rel", "noopener");
  }

  const closeMenu = () => {
    mobileMenu?.classList.remove("open");
    menuToggle?.setAttribute("aria-expanded", "false");
    menuToggle?.setAttribute("aria-label", "Open navigation");
    document.body.classList.remove("menu-open");
  };

  menuToggle?.addEventListener("click", () => {
    const open = mobileMenu?.classList.toggle("open");
    menuToggle.setAttribute("aria-expanded", String(Boolean(open)));
    menuToggle.setAttribute("aria-label", open ? "Close navigation" : "Open navigation");
    document.body.classList.toggle("menu-open", Boolean(open));
  });
  mobileMenu?.querySelectorAll("a").forEach((link) => link.addEventListener("click", closeMenu));

  const updateNav = () => nav?.classList.toggle("scrolled", window.scrollY > 16);
  updateNav();
  window.addEventListener("scroll", updateNav, { passive: true });

  const revealObserver = new IntersectionObserver((entries, observer) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("visible");
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.12, rootMargin: "0px 0px -35px" });
  document.querySelectorAll(".reveal").forEach((element) => revealObserver.observe(element));

  const navLinks = [...document.querySelectorAll(".nav-links a")];
  const sections = [...document.querySelectorAll("main section[id]")];
  const sectionObserver = new IntersectionObserver((entries) => {
    const visible = entries
      .filter((entry) => entry.isIntersecting)
      .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!visible) return;
    navLinks.forEach((link) => {
      link.classList.toggle("active", link.getAttribute("href") === `#${visible.target.id}`);
    });
  }, { rootMargin: "-25% 0px -62% 0px", threshold: [0, 0.1, 0.5] });
  sections.forEach((section) => sectionObserver.observe(section));

  const updatePlayControl = () => {
    playControl?.classList.toggle("paused", !playing);
    playControl?.setAttribute("aria-label", playing ? "Pause comparison videos" : "Play comparison videos");
    const label = playControl?.querySelector("span");
    if (label) label.textContent = playing ? "Pause" : "Play";
  };

  const pauseVideos = () => {
    videos.forEach((video) => video.pause());
    playing = false;
    updatePlayControl();
  };

  const playVideos = async (restart = false) => {
    if (!resultVisible) return;
    if (restart) videos.forEach((video) => { video.currentTime = 0; });
    await Promise.all(videos.map((video) => video.play().catch(() => null)));
    playing = videos.some((video) => !video.paused);
    updatePlayControl();
  };

  const toggleVideos = () => {
    if (playing) pauseVideos();
    else playVideos(false);
  };

  playControl?.addEventListener("click", toggleVideos);
  videos.forEach((video) => video.addEventListener("click", toggleVideos));
  updatePlayControl();

  sceneButtons.forEach((button) => {
    button.addEventListener("click", async () => {
      const scene = button.dataset.scene;
      const loadId = ++sceneLoadId;
      sceneButtons.forEach((item) => {
        const active = item === button;
        item.classList.toggle("active", active);
        item.setAttribute("aria-selected", String(active));
      });

      const ready = videos.map((video) => new Promise((resolve) => {
        video.addEventListener("canplay", resolve, { once: true });
        const source = video.querySelector("source");
        source.src = `assets/videos/${scene}_${video.dataset.method}.mp4`;
        video.load();
      }));
      await Promise.all(ready);
      if (loadId !== sceneLoadId) return;
      playVideos(true);
    });
  });

  const resultObserver = new IntersectionObserver((entries) => {
    resultVisible = entries[0].isIntersecting;
    if (resultVisible) playVideos(false);
    else pauseVideos();
  }, { threshold: 0.16 });
  const videoDemo = document.querySelector(".video-demo");
  if (videoDemo) resultObserver.observe(videoDemo);

  window.setInterval(() => {
    if (!playing || videos.length < 2) return;
    const referenceTime = videos[0].currentTime;
    videos.slice(1).forEach((video) => {
      if (Math.abs(video.currentTime - referenceTime) > 0.12) video.currentTime = referenceTime;
    });
  }, 1200);

  const copyButton = document.querySelector("#copy-bibtex");
  copyButton?.addEventListener("click", async () => {
    const citation = document.querySelector("#bibtex")?.textContent.trim() || "";
    try {
      await navigator.clipboard.writeText(citation);
    } catch {
      const textarea = document.createElement("textarea");
      textarea.value = citation;
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      textarea.remove();
    }
    const label = copyButton.querySelector("span");
    if (label) label.textContent = "Copied";
    window.setTimeout(() => {
      if (label) label.textContent = "Copy";
    }, 1800);
  });
})();
