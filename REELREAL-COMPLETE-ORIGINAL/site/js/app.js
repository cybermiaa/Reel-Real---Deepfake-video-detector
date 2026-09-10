/* ============================================================================
   REEL/REAL — WEBSITE UI CONTROLLER
   ----------------------------------------------------------------------------
   Website-only. Owns DOM wiring and rendering; owns no detection logic.

   The only contact with analysis is:
       ReelReal.analyzeVideo(file, { onProgress, forceVerdict })
   ...plus ReelReal.decodeResult() for reports handed over from the extension.
   ========================================================================== */

(function () {
  'use strict';

  var $ = function (sel) { return document.querySelector(sel); };

  /* ------------------------------------------------------------------------
     Hero video pair — hover to preview
     ---------------------------------------------------------------------- */
  document.querySelectorAll('.slot').forEach(function (slot) {
    var video = slot.querySelector('video');
    var ghost = slot.querySelector('.ghost');

    // Only reveal the video once a frame is actually decodable, otherwise a
    // missing asset would show as a black rectangle instead of the placeholder.
    video.addEventListener('loadeddata', function () { ghost.style.display = 'none'; });
    video.addEventListener('error', function () { video.style.display = 'none'; });

    slot.addEventListener('mouseenter', function () { video.play().catch(function () {}); });
    slot.addEventListener('mouseleave', function () { video.pause(); });
  });

  /* ------------------------------------------------------------------------
     Upload staging
     ---------------------------------------------------------------------- */
  var drop = $('#drop');
  var fileInput = $('#file');
  var staged = $('#staged');

  var currentFile = null;   // real File, or a {name,size} stand-in for samples
  var objectURL = null;     // blob: URL for playback; null for samples/imports
  var forcedVerdict = null; // demo-only override from the sample chips

  $('#pick').addEventListener('click', function () { fileInput.click(); });

  fileInput.addEventListener('change', function (e) {
    if (e.target.files[0]) stageFile(e.target.files[0]);
  });

  // preventDefault on dragover is what actually makes an element a drop target;
  // without it the browser just navigates to the dropped file.
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

  document.querySelectorAll('.chip').forEach(function (chip) {
    chip.addEventListener('click', function () {
      forcedVerdict = chip.dataset.fake === '1' ? 'synthetic' : 'authentic';
      // Samples are not real files — just enough shape for the detector.
      stageFile({ name: chip.dataset.sample, size: 14.7 * 1024 * 1024, __sample: true });
    });
  });

  function stageFile(file) {
    currentFile = file;

    if (file.__sample) {
      objectURL = null;                       // nothing to play back
    } else {
      forcedVerdict = null;                   // real files get a real verdict
      if (objectURL) URL.revokeObjectURL(objectURL);  // don't leak the previous blob
      objectURL = URL.createObjectURL(file);
    }

    $('#sName').textContent = file.name;
    $('#sMeta').textContent = ReelReal.formatSize(file.size) + ' · Ready';

    staged.classList.add('on');
    resetProgress();
    staged.scrollIntoView({ block: 'center', behavior: 'smooth' });
  }

  $('#clear').addEventListener('click', function () {
    staged.classList.remove('on');
    currentFile = null;
    forcedVerdict = null;
    if (objectURL) { URL.revokeObjectURL(objectURL); objectURL = null; }
    fileInput.value = '';
  });

  /* ------------------------------------------------------------------------
     Progress
     ---------------------------------------------------------------------- */
  var bar = $('#bar');
  var barFill = bar.querySelector('i');
  var barLabel = $('#barLabel');

  function resetProgress() {
    bar.classList.remove('on');
    barLabel.classList.remove('on');
    barFill.style.width = '0%';
  }

  /* ------------------------------------------------------------------------
     Run analysis
     ---------------------------------------------------------------------- */
  $('#run').addEventListener('click', async function () {
    if (!currentFile) return;

    bar.classList.add('on');
    barLabel.classList.add('on');
    $('#run').disabled = true;

    try {
      var result = await ReelReal.analyzeVideo(currentFile, {
        forceVerdict: forcedVerdict,
        onProgress: function (p) {
          barFill.style.width = p.percent + '%';
          barLabel.textContent = p.stage;
        }
      });
      renderReport(result, { playbackURL: objectURL });
    } catch (err) {
      barLabel.textContent = 'Analysis failed: ' + err.message;
    } finally {
      $('#run').disabled = false;
    }
  });

  /* ------------------------------------------------------------------------
     Report rendering
     --------------------------------------------------------------------------
     Pure function of (AnalysisResult, playbackURL). It never inspects the File,
     which is what lets an imported result from the extension render through the
     exact same path with playbackURL = null.
     ---------------------------------------------------------------------- */
  function renderReport(result, opts) {
    opts = opts || {};
    var isSynthetic = result.verdict === 'synthetic';

    $('#rName').textContent = result.fileName;
    $('#rMeta').textContent = [
      ReelReal.formatSize(result.fileSizeBytes),
      ReelReal.formatClock(result.durationSeconds),
      result.resolution,
      result.provenance.summary
    ].join(' · ');

    $('#rScore').textContent = result.confidence.toFixed(2);

    /* No band is produced until clip-level calibration has been fitted. Saying
       "Uncalibrated" is the honest rendering — a range here would assert an
       accuracy the model has not been checked for. */
    var band = result.calibratedBand;
    $('#rBand').textContent = band
      ? band.low.toFixed(2) + '–' + band.high.toFixed(2)
      : 'Uncalibrated';

    /* "none" is a finding and must not be shown when the model could not see
       enough of a face to look — that case gets a dash, not a clean answer. */
    var manipulated = result.manipulatedDurationSeconds;
    $('#rDur').textContent =
      result.verdict === 'authentic' ? 'none'
      : result.verdict === 'inconclusive' ? '—'
      : (typeof manipulated === 'number' ? manipulated.toFixed(1) + ' s' : '—');

    var badge = $('#rBadge');
    badge.textContent = isSynthetic ? 'Likely synthetic'
      : result.verdict === 'inconclusive' ? 'Insufficient evidence'
      : 'No manipulation found';
    /* "clean" is the green treatment. Inconclusive is not a pass, so it keeps
       the neutral/alert styling rather than being coloured like a clean bill. */
    badge.classList.toggle('clean', result.verdict === 'authentic');

    $('#stamp').textContent = 'Analysed in ' + (result.processingTimeMs / 1000).toFixed(1) +
                              's · model ' + result.modelVersion +
                              (result.isMock ? ' · SIMULATED RESULT' : '');

    // Evidence rows: the <b> labels stay as authored, only the findings change.
    result.artifacts.forEach(function (artifact) {
      var el = document.getElementById(artifact.id);
      if (!el) return;
      el.textContent = artifact.detail;
      el.className = artifact.severity;   // ok | warn | bad -> coloured by tokens.css
    });

    /* Video preview. Samples and extension imports have no playable blob. */
    var video = $('#rVideo');
    var ghost = $('#vGhost');
    if (opts.playbackURL) {
      video.src = opts.playbackURL;
      video.style.display = 'block';
      ghost.style.display = 'none';
    } else {
      video.removeAttribute('src');
      video.style.display = 'none';
      ghost.style.display = 'grid';
      ghost.textContent = opts.ghostText || 'Sample Clip — Preview Unavailable';
    }

    renderTimeline(result, video, Boolean(opts.playbackURL));

    $('#report').classList.add('on');
    $('#report').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function renderTimeline(result, video, canSeek) {
    var bars = $('#bars');
    bars.innerHTML = '';

    // One <button> per second. Buttons (not divs) so the timeline is keyboard
    // navigable and screen-reader announceable for free.
    result.timeline.forEach(function (point) {
      var b = document.createElement('button');
      b.style.height = Math.max(12, point.score * 100) + '%';
      b.className = point.label === 'synthetic' ? 'f' : (point.label === 'uncertain' ? 'w' : '');
      b.title = ReelReal.formatClock(point.second) + ' — score ' + point.score.toFixed(2);
      b.setAttribute('aria-label', 'Second ' + point.second + ', score ' + point.score.toFixed(2));

      b.addEventListener('click', function () {
        if (canSeek) {
          video.currentTime = point.second;
          video.play().catch(function () {});
        }
        markCurrent(point.second);
      });

      bars.appendChild(b);
    });

    $('#axisEnd').textContent = ReelReal.formatClock(result.durationSeconds);

    var seg = result.flaggedSegment;
    $('#axisMid').textContent = seg
      ? 'Flagged segment: ' + ReelReal.formatClock(seg.startSecond) + '–' + ReelReal.formatClock(seg.endSecond)
      : 'No flags';
    $('#axisMid').style.color = seg ? 'var(--fake)' : 'var(--text-muted)';

    function markCurrent(index) {
      Array.prototype.forEach.call(bars.children, function (el, n) {
        el.classList.toggle('now', n === index);
      });
    }

    // Keep the highlighted bar in sync while the clip plays.
    video.ontimeupdate = function () { markCurrent(Math.floor(video.currentTime)); };
  }

  /* ------------------------------------------------------------------------
     Report actions
     ---------------------------------------------------------------------- */
  $('#again').addEventListener('click', function () {
    $('#report').classList.remove('on');
    $('#handoffNote').classList.remove('on');
    staged.classList.remove('on');
    // Clear any handed-off result so a refresh doesn't resurrect it.
    if (location.hash) history.replaceState(null, '', location.pathname + location.search);
    document.getElementById('detect').scrollIntoView({ behavior: 'smooth' });
  });

  // The print stylesheet in site.css hides everything except the report, so the
  // browser's "Save as PDF" destination produces a clean one-page export.
  $('#exportPdf').addEventListener('click', function () { window.print(); });

  /* ------------------------------------------------------------------------
     Handoff from the Chrome extension
     --------------------------------------------------------------------------
     The popup opens this page as:  index.html#result=<base64url(JSON)>
     A fragment is chosen deliberately: it is never transmitted to a server, so
     the report stays as local as the extension's own analysis was.
     ---------------------------------------------------------------------- */
  function importHandoff() {
    var match = /[#&]result=([^&]+)/.exec(location.hash);
    if (!match) return;

    var result = ReelReal.decodeResult(match[1]);
    if (!result) return;

    var note = $('#handoffNote');
    note.textContent = 'Imported from the REEL/REAL extension · ' + result.fileName +
                       ' · the original video stays on your device, so playback is unavailable here.';
    note.classList.add('on');

    renderReport(result, {
      playbackURL: null,
      ghostText: 'Analysed in the extension — video not transferred'
    });
  }

  importHandoff();
})();
