/* ============================================================================
   REEL/REAL — DETECTION MODULE
   ----------------------------------------------------------------------------
   This file is the single source of analysis truth for BOTH surfaces:
     site/js/detector.js        <-- you are here (the master copy)
     extension/js/detector.js   <-- byte-identical duplicate

   THIS IS THE ONLY FILE THAT KNOWS WHERE VERDICTS COME FROM.

   Everything else in the project talks to exactly one function:

       ReelReal.analyzeVideo(file, options) -> Promise<AnalysisResult>

   It now uploads the video to the Python detection server (see server/app.py)
   and returns what that server produces. The original mocked generator is kept
   below and is used in exactly two situations:

     1. sample chips, which are filenames with no actual video behind them,
     2. the server being unreachable, so the site still demos offline.

   Either way the result carries isMock:true and the interface says so. A
   mocked result can never be silently mistaken for a real verdict.

   ----------------------------------------------------------------------------
   AnalysisResult contract
   ----------------------------------------------------------------------------
   {
     id: string,
     fileName: string,
     fileSizeBytes: number,
     durationSeconds: number,
     resolution: string,               // "720p", or "unknown"

     verdict: "synthetic" | "authentic" | "inconclusive",
     confidence: number,               // 0..1, probability the clip is synthetic

     // A 90% interval, or null when no calibration has been fitted — which is
     // the case today. Null makes the UI print "Uncalibrated" rather than
     // showing a range the model has not earned. See the calibration note in
     // server/README.md.
     calibratedBand: { low: number, high: number } | null,

     manipulatedDurationSeconds: number | null,       // null when underivable
     flaggedSegment: { startSecond, endSecond } | null,

     timeline: [                       // one entry per second of video
       { second: number, score: number, label: "authentic"|"uncertain"|"synthetic" }
     ],

     artifacts: [                      // forensic findings, ordered for display
       // severity "na" = this model does not measure it; rendered greyed out
       { id: string, label: string, detail: string, severity: "ok"|"warn"|"bad"|"na" }
     ],

     provenance: { signed: boolean, summary: string },

     processingTimeMs: number,
     modelVersion: string,
     analysedAt: string,               // ISO 8601

     isMock: boolean,                  // true = generated, not detected
     pipeline?: { ... }                // extra real measurements the UI has no
                                       // slot for yet (coverage, frame counts,
                                       // evidence sentences). Server only.
   }

   Note: `artifacts[].id` values e1..e6 line up with the six evidence rows that
   already exist in the site's markup, so app.js can fill them in place.
   ========================================================================== */

(function (global) {
  'use strict';

  var MODEL_VERSION = 'v2.4';

  /* Where the Python detection server is listening. Set
     window.REELREAL_API_BASE before this script loads to point both surfaces at
     a deployed backend without editing this file. */
  var API_BASE = global.REELREAL_API_BASE || 'http://127.0.0.1:8000';

  /* If the server cannot be reached, fall back to the mocked generator so the
     site is still demonstrable with no backend running. Set to false to make a
     missing server a hard error instead.
     Note this only covers "no answer at all". If the server answers with an
     error, that error is surfaced — a real failure is never papered over with
     an invented result. */
  var MOCK_FALLBACK = true;

  /* Score thresholds. Shared so the timeline colouring, the verdict and the
     legend can never disagree with each other. */
  var THRESHOLD_SYNTHETIC = 0.60;
  var THRESHOLD_UNCERTAIN = 0.35;

  /* The stages we surface while "analysing". A real pipeline would emit these
     from the server (or from local frame extraction) instead of a timer. */
  var STAGES = [
    'Uploading video…',
    'Sampling frames and detecting faces…',
    'Scoring face crops…',
    'Aggregating clip verdict…'
  ];

  /* --------------------------------------------------------------------------
     Deterministic pseudo-randomness
     --------------------------------------------------------------------------
     Seeded from the filename (FNV-1a hash -> mulberry32) so the same file
     always produces the same demo report. Without this, re-analysing the same
     clip would flip the verdict and the demo would feel broken.
     ------------------------------------------------------------------------ */
  function seededRandom(seedString) {
    var h = 2166136261;
    for (var i = 0; i < seedString.length; i++) {
      h ^= seedString.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return function () {
      h += 0x6D2B79F5;
      var t = h;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function sleep(ms) {
    return new Promise(function (resolve) { setTimeout(resolve, ms); });
  }

  function labelFor(score) {
    if (score >= THRESHOLD_SYNTHETIC) return 'synthetic';
    if (score >= THRESHOLD_UNCERTAIN) return 'uncertain';
    return 'authentic';
  }

  /* mm:ss */
  function formatClock(seconds) {
    return Math.floor(seconds / 60) + ':' + String(Math.floor(seconds % 60)).padStart(2, '0');
  }

  function formatSize(bytes) {
    return (bytes / 1048576).toFixed(1) + ' MB';
  }

  /* --------------------------------------------------------------------------
     Result synthesis
     --------------------------------------------------------------------------
     Builds a report that *looks* like real model output: a mostly-quiet
     baseline with a contiguous manipulated burst, soft shoulders either side of
     it, and a clip-level score derived from the peak rather than the mean.
     ------------------------------------------------------------------------ */
  function buildResult(file, forceVerdict, elapsedMs) {
    var rnd = seededRandom(file.name);
    var duration = 24;                       // fixed demo length

    var isSynthetic = (forceVerdict === 'synthetic') ? true
                    : (forceVerdict === 'authentic') ? false
                    : rnd() > 0.42;

    /* Where the tampering sits, when there is any. */
    var start = 5 + Math.floor(rnd() * 7);
    var span = 5 + Math.floor(rnd() * 4);

    var timeline = [];
    for (var i = 0; i < duration; i++) {
      var score = 0.06 + rnd() * 0.2;                       // baseline noise
      if (isSynthetic && i >= start && i < start + span) {
        score = 0.72 + rnd() * 0.26;                        // manipulated burst
      } else if (isSynthetic && (i === start - 1 || i === start + span)) {
        score = 0.42 + rnd() * 0.14;                        // blend shoulders
      }
      score = Math.min(0.99, score);
      timeline.push({ second: i, score: score, label: labelFor(score) });
    }

    /* Clip-level score: peak-driven when synthetic (a 5s fake inside a 24s clip
       is still a fake clip), otherwise a low baseline. A mean would dilute
       short edits into nothing — the exact failure mode called out in the
       "Micro-Edits" section on the site. */
    var peak = timeline.reduce(function (m, t) { return Math.max(m, t.score); }, 0);
    var confidence = isSynthetic ? peak - 0.03 : 0.08 + rnd() * 0.14;

    var low = Math.max(0.01, confidence - 0.06 - rnd() * 0.02);
    var high = Math.min(0.99, confidence + 0.04 + rnd() * 0.02);

    var artifacts = isSynthetic ? [
      { id: 'e1', label: 'Face boundary blending', detail: 'Soft seam along jawline', severity: 'bad' },
      { id: 'e2', label: 'Blink rate & frequency', detail: '2 blinks in ' + duration + ' s — unnatural', severity: 'warn' },
      { id: 'e3', label: 'Temporal flicker', detail: 'Texture reset at ' + (start + 0.1).toFixed(1) + ' s', severity: 'bad' },
      { id: 'e4', label: 'Compression trace', detail: 'Double-encoded region, ' + formatClock(start) + '–' + formatClock(start + span), severity: 'warn' },
      { id: 'e5', label: 'Lip-sync alignment', detail: 'Consistent', severity: 'ok' },
      { id: 'e6', label: 'C2PA Provenance', detail: 'Unsigned', severity: 'bad' }
    ] : [
      { id: 'e1', label: 'Face boundary blending', detail: 'No blending detected', severity: 'ok' },
      { id: 'e2', label: 'Blink rate & frequency', detail: '9 blinks in ' + duration + ' s — typical', severity: 'ok' },
      { id: 'e3', label: 'Temporal flicker', detail: 'Stable across all frames', severity: 'ok' },
      { id: 'e4', label: 'Compression trace', detail: 'Single encode, sensor noise intact', severity: 'ok' },
      { id: 'e5', label: 'Lip-sync alignment', detail: 'Consistent', severity: 'ok' },
      { id: 'e6', label: 'C2PA Provenance', detail: 'Signed by capture device', severity: 'ok' }
    ];

    return {
      id: 'an_' + Math.abs(Math.floor(rnd() * 1e9)).toString(36),
      fileName: file.name,
      fileSizeBytes: file.size,
      durationSeconds: duration,
      resolution: '720p',

      verdict: isSynthetic ? 'synthetic' : 'authentic',
      confidence: Number(confidence.toFixed(2)),
      calibratedBand: { low: Number(low.toFixed(2)), high: Number(high.toFixed(2)) },

      manipulatedDurationSeconds: isSynthetic ? Number((span + 0.2).toFixed(1)) : 0,
      flaggedSegment: isSynthetic ? { startSecond: start, endSecond: start + span } : null,

      timeline: timeline,
      artifacts: artifacts,
      provenance: {
        signed: !isSynthetic,
        summary: isSynthetic ? 'no C2PA signature' : 'C2PA signed'
      },

      processingTimeMs: elapsedMs,
      modelVersion: MODEL_VERSION,
      analysedAt: new Date().toISOString()
    };
  }

  /* --------------------------------------------------------------------------
     Upload to the detection server.
     --------------------------------------------------------------------------
     XMLHttpRequest rather than fetch() on purpose: fetch cannot report upload
     progress, and on a phone-shot video the upload is the slow part. XHR fires
     real byte-level progress events.
     ------------------------------------------------------------------------ */
  function postToServer(file, onProgress, signal) {
    return new Promise(function (resolve, reject) {
      var xhr = new XMLHttpRequest();
      var body = new FormData();
      body.append('video', file, file.name);

      xhr.open('POST', API_BASE + '/v1/analyze', true);
      xhr.responseType = 'json';
      xhr.timeout = 15 * 60 * 1000;    // first request downloads model weights

      var creep = null;
      function stopCreep() {
        if (creep) { clearInterval(creep); creep = null; }
      }

      /* The upload is genuinely measurable, so the bar tracks real bytes — but
         only up to 60%. The remaining 40% is the model working, and the server
         stays silent until it has an answer. */
      xhr.upload.onprogress = function (e) {
        if (!e.lengthComputable) return;
        onProgress({ percent: (e.loaded / e.total) * 60, stage: STAGES[0] });
      };

      /* Bytes are all sent; the model is now running. There are no further
         events to listen to, so the bar creeps towards 95% to show the request
         is still alive. It deliberately never reaches 100 on a guess — only a
         real response completes it. */
      xhr.upload.onload = function () {
        var pct = 60;
        creep = setInterval(function () {
          pct = Math.min(95, pct + 1.2);
          var i = Math.min(STAGES.length - 1, 1 + Math.floor((pct - 60) / 14));
          onProgress({ percent: pct, stage: STAGES[i] });
        }, 400);
      };

      xhr.onload = function () {
        stopCreep();
        if (xhr.status >= 200 && xhr.status < 300) {
          onProgress({ percent: 100, stage: STAGES[STAGES.length - 1] });
          resolve(xhr.response);
          return;
        }
        var detail = (xhr.response && xhr.response.detail) || ('HTTP ' + xhr.status);
        var err = new Error(String(detail));
        /* The server answered — it just answered badly. Marked so the caller
           does NOT quietly substitute a mocked result for a real failure. */
        err.serverReached = true;
        reject(err);
      };

      xhr.onerror = function () {
        stopCreep();
        reject(new Error('Cannot reach the detection server at ' + API_BASE));
      };
      xhr.ontimeout = function () {
        stopCreep();
        var err = new Error('Analysis timed out.');
        err.serverReached = true;
        reject(err);
      };
      xhr.onabort = function () {
        stopCreep();
        reject(new DOMException('Analysis cancelled', 'AbortError'));
      };

      if (signal) {
        if (signal.aborted) { xhr.abort(); return; }
        signal.addEventListener('abort', function () { xhr.abort(); });
      }

      xhr.send(body);
    });
  }

  /* ==========================================================================
     THE SEAM
     ==========================================================================
     @param {File|{name,size}} file    A File from an <input> or a drop event.
     @param {Object}  [options]
     @param {Function} [options.onProgress]  ({percent, stage}) => void
     @param {AbortSignal} [options.signal]   Cancels an in-flight analysis.
     @param {"synthetic"|"authentic"} [options.forceVerdict]
            DEMO ONLY — lets the sample chips guarantee a given outcome.
            Ignored for real uploads.
     @returns {Promise<AnalysisResult>}
     ======================================================================== */
  async function analyzeVideo(file, options) {
    options = options || {};
    var onProgress = options.onProgress || function () {};
    var signal = options.signal;
    var startedAt = Date.now();

    if (!file) throw new Error('No file provided');

    /* Sample chips are {name, size} placeholders with no bytes behind them.
       There is nothing to upload, so they stay mocked. */
    var hasBytes = (typeof Blob !== 'undefined') && (file instanceof Blob);

    if (hasBytes) {
      try {
        var real = await postToServer(file, onProgress, signal);
        real.isMock = false;
        return real;
      } catch (err) {
        if (err && err.name === 'AbortError') throw err;
        /* Server answered with an error, or the user asked for hard failures:
           surface it. Only a completely unreachable server falls through. */
        if (err.serverReached || !MOCK_FALLBACK) throw err;
        console.warn('[REEL/REAL] ' + err.message + ' — falling back to a MOCKED result.');
      }
    }

    /* ----- mocked path (samples, or no server) ----- */
    var percent = 0;
    while (percent < 100) {
      if (signal && signal.aborted) throw new DOMException('Analysis cancelled', 'AbortError');

      percent = Math.min(100, percent + Math.random() * 11 + 5);
      var stageIndex = Math.min(STAGES.length - 1, Math.floor(percent / 26));
      onProgress({ percent: percent, stage: STAGES[stageIndex] });

      await sleep(170);
    }

    /* Small settle before the report appears, so the bar visibly completes. */
    await sleep(320);

    var mocked = buildResult(file, options.forceVerdict, Date.now() - startedAt);
    mocked.isMock = true;
    return mocked;
  }

  /* ==========================================================================
     HANDOFF ENCODING
     ==========================================================================
     The extension popup and the website are separate origins with separate
     storage, so a result travels between them as a URL fragment:

         <site>/index.html#result=<base64url(JSON)>

     A fragment never leaves the browser (it is not sent to any server), and
     both surfaces use the identical pair of functions below, which is exactly
     why this lives in the shared module.

     The video file itself cannot travel this way — see the write-up.
     ======================================================================== */

  function encodeResult(result) {
    // btoa() is byte-oriented, so UTF-8 encode first or any non-ASCII filename throws.
    var utf8 = encodeURIComponent(JSON.stringify(result)).replace(
      /%([0-9A-F]{2})/g,
      function (_, hex) { return String.fromCharCode(parseInt(hex, 16)); }
    );
    // base64url: +/ are legal in a fragment but get mangled by copy/paste and
    // some link parsers, so swap them out and drop the padding.
    return btoa(utf8).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
  }

  function decodeResult(encoded) {
    try {
      var b64 = encoded.replace(/-/g, '+').replace(/_/g, '/');
      while (b64.length % 4) b64 += '=';
      var bytes = atob(b64);
      var utf8 = Array.prototype.map.call(bytes, function (c) {
        return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
      }).join('');
      return JSON.parse(decodeURIComponent(utf8));
    } catch (err) {
      console.warn('[REEL/REAL] Could not decode handed-off result:', err);
      return null;
    }
  }

  /* Exposed as a global rather than an ES module on purpose: ES module imports
     are blocked on file:// URLs, and this project should run by double-clicking
     index.html with no build step and no local server. */
  global.ReelReal = {
    analyzeVideo: analyzeVideo,
    encodeResult: encodeResult,
    decodeResult: decodeResult,
    formatClock: formatClock,
    formatSize: formatSize,
    labelFor: labelFor,
    MODEL_VERSION: MODEL_VERSION,
    THRESHOLD_SYNTHETIC: THRESHOLD_SYNTHETIC,
    THRESHOLD_UNCERTAIN: THRESHOLD_UNCERTAIN
  };
})(typeof self !== 'undefined' ? self : this);
