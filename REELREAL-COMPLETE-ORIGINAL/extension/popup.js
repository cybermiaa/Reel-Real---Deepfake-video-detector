/* ============================================================================
   REEL/REAL — POPUP UI CONTROLLER
   ----------------------------------------------------------------------------
   Extension-only. The popup's counterpart to the website's js/app.js: same job,
   same shared detector, completely different layout.

   Two things make this different from an ordinary web page script:

   1. The popup is DESTROYED the moment it loses focus. Every variable here is
      gone when the user clicks away. Anything that must survive goes into
      chrome.storage (see saveResult / restoreResult below).

   2. There is no inline script allowed and no eval — Manifest V3's content
      security policy. Hence this external file and zero onclick="" attributes.
   ========================================================================== */

(function () {
  'use strict';

  /* ==========================================================================
     CONFIG — where the "Open full report on the website" link goes.
     ==========================================================================
     Right now this points at the site folder sitting on this machine, so the
     whole thing runs with no server at all. For this to work you must tick
     "Allow access to file URLs" on the extension's details page in Chrome —
     extensions cannot open local files without that checkbox.

     When the site is actually deployed, change this to the real address, e.g.
         var WEBSITE_URL = 'https://reelreal.example.com/';
     and the checkbox stops mattering.
     ======================================================================== */
  var WEBSITE_URL = 'file:///C:/Users/Zobia/Desktop/DEEPFAKE-PROJ/site/index.html';

  var STORAGE_KEY = 'lastResult';

  var $ = function (sel) { return document.querySelector(sel); };

  /* chrome.* is undefined if you open popup.html directly in a normal tab for
     design work. Everything still runs; only persistence and tab-opening
     degrade to web equivalents. */
  var hasChrome = typeof chrome !== 'undefined' && chrome.storage && chrome.storage.session;

  var currentFile = null;
  var forcedVerdict = null;
  var lastResult = null;

  /* ------------------------------------------------------------------------
     Screen switching
     ---------------------------------------------------------------------- */
  function showScreen(id) {
    document.querySelectorAll('.screen').forEach(function (s) {
      s.classList.toggle('on', s.id === id);
    });
    $('.pop-body').scrollTop = 0;
  }

  /* ------------------------------------------------------------------------
     Screen 1 — file selection
     ---------------------------------------------------------------------- */
  var drop = $('#drop');
  var fileInput = $('#file');

  $('#pick').addEventListener('click', function () { fileInput.click(); });

  fileInput.addEventListener('change', function (e) {
    if (e.target.files[0]) stageFile(e.target.files[0]);
  });

  ['dragenter', 'dragover'].forEach(function (type) {
    drop.addEventListener(type, function (e) { e.preventDefault(); drop.classList.add('over'); });
  });
  ['dragleave', 'drop'].forEach(function (type) {
    drop.addEventListener(type, function (e) { e.preventDefault(); drop.classList.remove('over'); });
  });
  drop.addEventListener('drop', function (e) {
    var file = e.dataTransfer.files[0];
    if (file) stageFile(file);
  });

  // Dropping a file anywhere else in a popup would otherwise navigate the popup
  // to that file and blow the UI away entirely.
  window.addEventListener('dragover', function (e) { e.preventDefault(); });
  window.addEventListener('drop', function (e) { e.preventDefault(); });

  document.querySelectorAll('.chip').forEach(function (chip) {
    chip.addEventListener('click', function () {
      forcedVerdict = chip.dataset.fake === '1' ? 'synthetic' : 'authentic';
      stageFile({ name: chip.dataset.sample, size: 14.7 * 1024 * 1024, __sample: true });
    });
  });

  function stageFile(file) {
    currentFile = file;
    if (!file.__sample) forcedVerdict = null;

    $('#sName').textContent = file.name;
    $('#sMeta').textContent = ReelReal.formatSize(file.size) + ' · Ready';

    $('#bar').classList.remove('on');
    $('#barLabel').classList.remove('on');
    $('#bar').querySelector('i').style.width = '0%';
    $('#run').disabled = false;

    showScreen('screenStaged');
  }

  $('#clear').addEventListener('click', function () {
    currentFile = null;
    forcedVerdict = null;
    fileInput.value = '';
    showScreen('screenUpload');
  });

  /* ------------------------------------------------------------------------
     Screen 2 — run the analysis
     ---------------------------------------------------------------------- */
  $('#run').addEventListener('click', async function () {
    if (!currentFile) return;

    var bar = $('#bar');
    var fill = bar.querySelector('i');
    var label = $('#barLabel');

    bar.classList.add('on');
    label.classList.add('on');
    $('#run').disabled = true;
    $('#clear').disabled = true;

    try {
      var result = await ReelReal.analyzeVideo(currentFile, {
        forceVerdict: forcedVerdict,
        onProgress: function (p) {
          fill.style.width = p.percent + '%';
          label.textContent = p.stage;
        }
      });
      lastResult = result;
      saveResult(result);
      renderResult(result);
    } catch (err) {
      label.textContent = 'Analysis failed: ' + err.message;
      $('#run').disabled = false;
    } finally {
      $('#clear').disabled = false;
    }
  });

  /* ------------------------------------------------------------------------
     Screen 3 — render
     ---------------------------------------------------------------------- */
  function renderResult(result) {
    var isSynthetic = result.verdict === 'synthetic';

    var badge = $('#rBadge');
    badge.textContent = isSynthetic ? 'Likely synthetic'
      : result.verdict === 'inconclusive' ? 'Insufficient evidence'
      : 'No manipulation found';
    // Only a genuine pass gets the green treatment. Inconclusive is not a pass.
    badge.classList.toggle('clean', result.verdict === 'authentic');

    $('#rName').textContent = result.fileName;
    $('#rMeta').textContent = [
      ReelReal.formatSize(result.fileSizeBytes),
      ReelReal.formatClock(result.durationSeconds),
      result.resolution
    ].join(' · ');

    $('#rScore').textContent = result.confidence.toFixed(2);

    // No band exists until clip-level calibration is fitted; say so rather than
    // printing a range the model has not been checked against.
    var band = result.calibratedBand;
    $('#rBand').textContent = band
      ? band.low.toFixed(2) + '–' + band.high.toFixed(2)
      : 'Uncal.';

    var manipulated = result.manipulatedDurationSeconds;
    $('#rDur').textContent =
      result.verdict === 'authentic' ? 'none'
      : result.verdict === 'inconclusive' ? '—'
      : (typeof manipulated === 'number' ? manipulated.toFixed(1) + 's' : '—');

    // Timeline — read-only in the popup, there is no video element to seek.
    var bars = $('#bars');
    bars.innerHTML = '';
    result.timeline.forEach(function (point) {
      var b = document.createElement('button');
      b.type = 'button';
      b.disabled = true;
      b.style.height = Math.max(10, point.score * 100) + '%';
      b.className = point.label === 'synthetic' ? 'f' : (point.label === 'uncertain' ? 'w' : '');
      b.title = ReelReal.formatClock(point.second) + ' — ' + point.score.toFixed(2);
      bars.appendChild(b);
    });

    $('#axisEnd').textContent = ReelReal.formatClock(result.durationSeconds);
    var seg = result.flaggedSegment;
    $('#axisMid').textContent = seg
      ? ReelReal.formatClock(seg.startSecond) + '–' + ReelReal.formatClock(seg.endSecond)
      : 'No flags';
    $('#axisMid').style.color = seg ? 'var(--fake)' : 'var(--text-muted)';

    // Evidence rows, built from the same artifacts array the site consumes.
    var evidence = $('#evidence');
    evidence.innerHTML = '';
    result.artifacts.forEach(function (artifact) {
      var row = document.createElement('div');
      row.className = 'ev';

      var label = document.createElement('b');
      label.textContent = artifact.label;

      var detail = document.createElement('span');
      detail.className = artifact.severity;
      detail.textContent = artifact.detail;

      row.appendChild(label);
      row.appendChild(detail);
      evidence.appendChild(row);
    });

    $('#ver').textContent = 'model ' + result.modelVersion +
                            ' · ' + (result.processingTimeMs / 1000).toFixed(1) + 's' +
                            (result.isMock ? ' · SIMULATED' : '');

    showScreen('screenResult');
  }

  $('#again').addEventListener('click', function () {
    currentFile = null;
    forcedVerdict = null;
    lastResult = null;
    fileInput.value = '';
    clearSavedResult();
    showScreen('screenUpload');
  });

  /* ------------------------------------------------------------------------
     Persistence — why the "storage" permission exists
     --------------------------------------------------------------------------
     Closing the popup tears down this whole page. Without this, a user who
     analysed a clip and then clicked back into the page would reopen the popup
     to an empty upload screen and assume the analysis was lost.

     storage.session (not .local) keeps it in memory and wipes it when the
     browser closes, so a filename never lands on disk.
     ---------------------------------------------------------------------- */
  function saveResult(result) {
    if (!hasChrome) return;
    chrome.storage.session.set({ lastResult: result });
  }

  function clearSavedResult() {
    if (!hasChrome) return;
    chrome.storage.session.remove(STORAGE_KEY);
  }

  function restoreResult() {
    if (!hasChrome) return;
    chrome.storage.session.get(STORAGE_KEY, function (data) {
      if (data && data[STORAGE_KEY]) {
        lastResult = data[STORAGE_KEY];
        renderResult(lastResult);
      }
    });
  }

  /* ------------------------------------------------------------------------
     Handoff to the website
     --------------------------------------------------------------------------
     The result rides along as a URL fragment. Fragments are never sent to a
     server, so the report does not become someone else's data just because the
     user wanted a bigger view of it.

     The VIDEO cannot come with it. The file lives in this popup's memory as a
     File object; the website is a different origin with no access to it, and a
     video is far too large for a URL anyway. The site therefore renders the
     imported report with the preview pane replaced by an explanatory panel.
     ---------------------------------------------------------------------- */
  $('#openSite').addEventListener('click', function (e) {
    e.preventDefault();

    var url = WEBSITE_URL;
    if (lastResult) url += '#result=' + ReelReal.encodeResult(lastResult);

    if (typeof chrome !== 'undefined' && chrome.tabs) {
      // Note: chrome.tabs.create needs NO permission. Reading a tab's url or
      // title would need "tabs" — we deliberately never do that.
      chrome.tabs.create({ url: url });
      window.close();
    } else {
      window.open(url, '_blank');
    }
  });

  restoreResult();
})();
