/* ============================================================================
   REEL/REAL — SCROLL-REVEAL
   ----------------------------------------------------------------------------
   Purely cosmetic and fully independent of detector.js / app.js: it just
   toggles the .rr-in class (defined in css/animate.css) on headings and cards
   as they enter the viewport. Safe to remove without affecting app logic.
   ========================================================================== */
(function () {
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  var targets = document.querySelectorAll(
    '.sec-head, .drop h3, .method-grid .cell, .stat, .ev'
  );
  if (!('IntersectionObserver' in window) || !targets.length) {
    targets.forEach(function (el) { el.classList.add('rr-in'); });
    return;
  }

  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry, i) {
      if (entry.isIntersecting) {
        // small stagger for groups of cards that reveal together
        setTimeout(function () { entry.target.classList.add('rr-in'); }, i * 60);
        io.unobserve(entry.target);
      }
    });
  }, { threshold: 0.2, rootMargin: '0px 0px -40px 0px' });

  targets.forEach(function (el) { io.observe(el); });

  // The report section starts hidden (display:none) until app.js reveals it,
  // so its heading/cards need their own observer once .report gets .on.
  var report = document.getElementById('report');
  if (report) {
    var reportTargets = report.querySelectorAll('.sec-head, .stat, .ev');
    var reportIO = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry, i) {
        if (entry.isIntersecting) {
          setTimeout(function () { entry.target.classList.add('rr-in'); }, i * 60);
          reportIO.unobserve(entry.target);
        }
      });
    }, { threshold: 0.15 });

    var mo = new MutationObserver(function () {
      if (report.classList.contains('on')) {
        reportTargets.forEach(function (el) { reportIO.observe(el); });
      }
    });
    mo.observe(report, { attributes: true, attributeFilter: ['class'] });
  }
})();
