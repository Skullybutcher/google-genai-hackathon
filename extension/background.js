// background.js

// 1. Create the context menu item when the extension is installed.
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "analyzeImageWithTruthGuard",
    title: "Analyze Image with TruthGuard AI",
    contexts: ["image"] // This makes the option appear only when you right-click an image
  });
});

// 2. Listen for a click on our context menu item.
chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "analyzeImageWithTruthGuard") {
    // 3. Get the URL of the image that was clicked.
    const imageUrl = info.srcUrl;
    
    // 4. Open our popup.html in a new tab and pass the image URL to it.
    
    const analysisUrl = chrome.runtime.getURL(`popup.html?image=${encodeURIComponent(imageUrl)}`);
    chrome.tabs.create({ url: analysisUrl });
  }
});

// 3. Listen for messages from the React web app (for authentication)
chrome.runtime.onMessageExternal.addListener((request, sender, sendResponse) => {
  // Only accept messages from our Firebase web app
  const allowedOrigins = [
    'https://gen-lang-client-0860021451.web.app',
    'https://gen-lang-client-0860021451.firebaseapp.com'
  ];
  
  if (!allowedOrigins.includes(new URL(sender.url).origin)) {
    console.error('Unauthorized sender:', sender.url);
    return;
  }
  
  if (request.type === 'FIREBASE_AUTH_TOKEN') {
    // Save the token securely in extension storage
    chrome.storage.local.set({ 
      firebaseAuthToken: request.token,
      tokenTimestamp: Date.now()
    }, () => {
      console.log('Firebase auth token saved to extension storage');
      sendResponse({ success: true });
    });
    return true; // Keep the message channel open for async response
  }
  
  if (request.type === 'FIREBASE_LOGOUT') {
    // Clear the token when user logs out
    chrome.storage.local.remove(['firebaseAuthToken', 'tokenTimestamp'], () => {
      console.log('Firebase auth token cleared from extension storage');
      sendResponse({ success: true });
    });
    return true;
  }
});