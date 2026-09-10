/* ============================================================================
   REEL/REAL — VISUAL ENHANCEMENT CONTROLLER
   ----------------------------------------------------------------------------
   Purely cosmetic and fully independent of detector.js / app.js / reveal.js.
   It never reads or writes analysis state — it only:
     1. injects a few fixed, pointer-events:none decorative nodes for the
        ambient background (css/enhance.css draws them),
     2. wraps the hero headline's words in spans so they can stagger in,
     3. adds pointer-reactive tilt/sheen to cards and a light magnetic pull
        to buttons,
     4. watches for content app.js already renders (timeline bars, stat
        numbers) and gives it a nicer entrance — without changing the
        values themselves.
   Safe to remove without affecting any site functionality.
   ========================================================================== */
(function () {
  'use strict';

  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  /* ------------------------------------------------------------------------
     1. Ambient background — blobs, grain, scan bar, cursor glow
     ---------------------------------------------------------------------- */
  var atmo = document.createElement('div');
  atmo.id = 'rr-atmosphere';
  atmo.setAttribute('aria-hidden', 'true');
  ['b1', 'b2', 'b3', 'b4'].forEach(function (cls) {
    var blob = document.createElement('div');
    blob.className = 'rr-blob ' + cls;
    atmo.appendChild(blob);
  });
  document.body.prepend(atmo);

  var grain = document.createElement('div');
  grain.id = 'rr-grain';
  grain.setAttribute('aria-hidden', 'true');
  document.body.prepend(grain);

  var heroSection = document.querySelector('.hero');
  if (heroSection) {
    var grid = document.createElement('div');
    grid.id = 'rr-grid';
    grid.setAttribute('aria-hidden', 'true');
    heroSection.prepend(grid);
  }

  var isFinePointer = window.matchMedia('(pointer: fine)').matches;
  if (isFinePointer) {
    var glow = document.createElement('div');
    glow.id = 'rr-cursor-glow';
    glow.setAttribute('aria-hidden', 'true');
    document.body.appendChild(glow);

    var glowTimer = null;
    document.addEventListener('pointermove', function (e) {
      document.documentElement.style.setProperty('--rr-x', e.clientX + 'px');
      document.documentElement.style.setProperty('--rr-y', e.clientY + 'px');
      glow.classList.add('on');
      clearTimeout(glowTimer);
      glowTimer = setTimeout(function () { glow.classList.remove('on'); }, 1400);
    }, { passive: true });
  }

  /* Track whether we're currently over the dark section so the cursor glow
     (and nothing else — no layout, no colour tokens) can flip tint. */
  var darkSection = document.querySelector('.dark-theme-section');
  if (darkSection && 'IntersectionObserver' in window) {
    var darkIO = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        document.body.classList.toggle('rr-on-dark', entry.isIntersecting);
      });
    }, { threshold: 0.4 });
    darkIO.observe(darkSection);
  }

  /* ------------------------------------------------------------------------
     2. Hero headline — split the display words into staggered spans.
        Only the plain-text portion before the italic tagline <span> is
        touched, so the shimmering gradient tagline is left completely
        intact as one unit.
     ---------------------------------------------------------------------- */
  var title = document.querySelector('.title');
  if (title) {
    var firstText = title.firstChild;
    if (firstText && firstText.nodeType === Node.TEXT_NODE) {
      var words = firstText.textContent.trim().split(/\s+/);
      var frag = document.createDocumentFragment();
      words.forEach(function (word, i) {
        var span = document.createElement('b');
        span.className = 'rr-word';
        span.style.setProperty('--i', i);
        span.textContent = word;
        frag.appendChild(span);
        frag.appendChild(document.createTextNode(' '));
      });
      title.replaceChild(frag, firstText);
    }
  }

  /* ------------------------------------------------------------------------
     3. Pointer-reactive sheen for cards (tilt removed per request), and a
        light magnetic pull for primary buttons. Transform-only, so nothing
        reflows.
     ---------------------------------------------------------------------- */
  if (isFinePointer) {
    var tiltTargets = document.querySelectorAll('.cell, .glass-panel');
    tiltTargets.forEach(function (el) {
      el.addEventListener('pointermove', function (e) {
        var r = el.getBoundingClientRect();
        var px = (e.clientX - r.left) / r.width;   // 0..1
        var py = (e.clientY - r.top) / r.height;   // 0..1
        el.style.setProperty('--mx', (px * 100).toFixed(1) + '%');
        el.style.setProperty('--my', (py * 100).toFixed(1) + '%');
      });
    });

    var magnetTargets = document.querySelectorAll('.btn:not(.ghost)');
    magnetTargets.forEach(function (el) {
      el.addEventListener('pointermove', function (e) {
        var r = el.getBoundingClientRect();
        var mx = (e.clientX - r.left - r.width / 2) * 0.18;
        var my = (e.clientY - r.top - r.height / 2) * 0.35;
        el.style.transform = 'translate(' + mx.toFixed(1) + 'px,' + (my - 2).toFixed(1) + 'px)';
      });
      el.addEventListener('pointerleave', function () { el.style.transform = ''; });
    });
  }

  /* ------------------------------------------------------------------------
     4a. Timeline bars: app.js sets each bar's final height inline the
         instant it's created. To get a grow-in instead of a pop-in, we
         briefly hold new bars at 0 and release them on the next frame —
         purely visual, never touches the values app.js computed.
     ---------------------------------------------------------------------- */
  var bars = document.getElementById('bars');
  if (bars && 'MutationObserver' in window) {
    var barsMO = new MutationObserver(function (mutations) {
      mutations.forEach(function (m) {
        m.addedNodes.forEach(function (node) {
          if (node.nodeType !== 1 || node.tagName !== 'BUTTON') return;
          var target = node.style.height;
          node.style.height = '0%';
          node.style.opacity = '0';
          requestAnimationFrame(function () {
            requestAnimationFrame(function () {
              node.style.height = target;
              node.style.opacity = '';
            });
          });
        });
      });
    });
    barsMO.observe(bars, { childList: true });
  }

  /* ------------------------------------------------------------------------
     4b. Stat / score numbers: give freshly written values a soft pop by
         toggling a short-lived class whenever app.js changes their text.
     ---------------------------------------------------------------------- */
  var numberEls = document.querySelectorAll('.stat strong, #rScore, #rBand, #rDur');
  if (numberEls.length && 'MutationObserver' in window) {
    numberEls.forEach(function (el) {
      var numMO = new MutationObserver(function () {
        el.classList.remove('rr-pop');
        // force reflow so the animation can restart if triggered twice in a row
        void el.offsetWidth;
        el.classList.add('rr-pop');
      });
      numMO.observe(el, { characterData: true, childList: true, subtree: true });
    });
  }
})();
