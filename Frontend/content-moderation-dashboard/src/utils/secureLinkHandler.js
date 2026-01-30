// secureLinkHandler.js
import CryptoJS from 'crypto-js';

class SecureLinkHandler {
  static _separator = ':::';

  static _encryptData(type, id) {
    try {
      // Use Flutter's fallback method directly
      // Flutter: base64UrlEncode(utf8.encode('$type$_separator$id'))
      const fallbackData = `${type}${this._separator}${id}`;

      // Convert string to bytes then to base64 (matching Flutter's utf8.encode)
      const fallbackBytes = CryptoJS.enc.Utf8.parse(fallbackData);
      const fallbackBase64 = CryptoJS.enc.Base64.stringify(fallbackBytes);

      // Make URL safe (matching Flutter's base64UrlEncode)
      return fallbackBase64
        .replace(/\+/g, '-')
        .replace(/\//g, '_')
        .replace(/=/g, ''); // Remove padding

    } catch (e) {
      console.error('Fallback encryption error:', e instanceof Error ? e.message : String(e));
      // If even fallback fails, return basic encoding
      const basicData = `${type}${this._separator}${id}`;
      return btoa(basicData)
        .replace(/\+/g, '-')
        .replace(/\//g, '_')
        .replace(/=/g, '');
    }
  }

  static generateSnipLink(postId) {
    const encryptedId = this._encryptData('snip', postId.toString());
    return `https://www.bigshorts.co/home/snips?id=${encryptedId}`;
  }
}

export default SecureLinkHandler;