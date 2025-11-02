# TruthGuard Extension Setup Guide

## Prerequisites
1. The TruthGuard web app must be deployed and accessible
2. The Chrome extension must be installed in your browser

## Setup Steps

### Step 1: Install the Extension
1. Open Chrome and go to `chrome://extensions`
2. Enable "Developer mode"
3. Click "Load unpacked" and select the `extension` folder
4. Note the Extension ID shown at the top of the extension details (e.g., `abc123def456...`)

### Step 2: Link Extension to Web App
1. Open the TruthGuard web app in Chrome
2. Open the browser console (F12)
3. Run the following command, replacing `YOUR_EXTENSION_ID` with the ID from Step 1:

```javascript
localStorage.setItem('truthguard_extension_id', 'YOUR_EXTENSION_ID');
```

For example:
```javascript
localStorage.setItem('truthguard_extension_id', 'abc123def456ghijk789');
```

4. Reload the web app page

### Step 3: Log In to the Web App
1. Log in to the TruthGuard web app with your credentials
2. The authentication token will automatically be sent to the extension
3. Check the console for confirmation: "✅ Token sent to extension"

### Step 4: Test the Extension
1. Visit any news article on the web
2. Click the TruthGuard extension icon
3. The extension should now be able to analyze content using your authenticated session

## Troubleshooting

### "No authentication token found"
- Make sure you've logged in to the web app
- Try logging out and logging back in
- Check that the extension ID is correctly stored in localStorage

### "Extension not found or not responding"
- Verify the extension ID in localStorage is correct
- Make sure the extension is enabled in Chrome
- Try reloading the extension: go to `chrome://extensions` → click the reload icon

### Extension doesn't receive tokens
- Check that the web app domain is in the `externally_connectable` list in `manifest.json`
- Ensure both the web app and extension are using HTTPS (or localhost for development)

## Architecture Overview

The authentication flow works as follows:

1. **User logs in** to the web app → Firebase Auth generates an ID token
2. **Token is sent** from the web app to the extension via Chrome message passing
3. **Extension stores** the token securely in `chrome.storage.local`
4. **Extension uses** the token to authenticate API requests with `Authorization: Bearer <token>`

This architecture provides:
- ✅ Single sign-on experience
- ✅ Secure token storage in extension
- ✅ Automatic token updates when user logs in/out
- ✅ No need to manage separate extension credentials


