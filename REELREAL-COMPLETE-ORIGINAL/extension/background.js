/* ============================================================================
   REEL/REAL — BACKGROUND SERVICE WORKER
   ----------------------------------------------------------------------------
   A Manifest V3 service worker is NOT a page. It has no DOM, no window, no
   document, and Chrome shuts it down after roughly 30 seconds of inactivity,
   restarting it when an event it listens for fires. That means:

     - You cannot keep state in a module-level variable and expect it to survive.
       Anything durable goes in chrome.storage.
     - Every listener must be registered synchronously at the top level, on
       every startup. Registering one inside a callback means Chrome does not
       know to wake the worker for that event.

   Right now this file does almost nothing, and that is correct: with the
   analysis mocked and running inside the popup, there is no background work to
   do. It earns its keep once a real backend exists — see the note at the
   bottom.
   ========================================================================== */

/* Fires on first install and on every extension update. Good place for
   one-time setup; it must be registered at the top level to work at all. */
chrome.runtime.onInstalled.addListener(function (details) {
  if (details.reason === 'install') {
    console.log('[REEL/REAL] Installed. Open the toolbar icon to analyse a clip.');
  }
});

/* ============================================================================
   WHAT MOVES IN HERE WHEN THE BACKEND IS REAL
   ----------------------------------------------------------------------------
   Uploading a 200 MB video takes longer than a user will keep a popup open, and
   the popup is destroyed the instant it loses focus — which would abort the
   fetch mid-upload.

   So the flow becomes:

     popup  --chrome.runtime.sendMessage({type:'ANALYZE', ...})-->  worker
     worker --fetch(API/v1/analyze)-------------------------------> server
     worker --chrome.storage.session.set({job})------------------->  (state)
     popup  --reads storage on open, subscribes to storage.onChanged

   The worker survives the popup closing, writes progress and the final result
   to storage, and the popup simply reflects whatever storage says whenever it
   happens to be open. Add chrome.notifications if you want to tell the user a
   long job finished while the popup was shut.

   That change also needs manifest edits:
     "host_permissions": ["https://api.your-domain.com/*"]   to allow the fetch
     "permissions": ["storage", "notifications"]             if you notify
   ========================================================================== */
