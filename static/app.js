/**
 * ARA-1 Financial Research Agent — Chat Frontend
 *
 * Handles:
 *  - Sending messages to /api/chat
 *  - Rendering user & agent message bubbles
 *  - Typing indicator while agent processes
 *  - PDF download links inline
 *  - Lightweight Markdown → HTML conversion
 *  - Suggestion chip click handling
 *  - Auto-resize textarea
 */

// ─── DOM References ──────────────────────────────────────────
const chatMessages  = document.getElementById('chat-messages');
const welcomeScreen = document.getElementById('welcome-screen');
const chatInput     = document.getElementById('chat-input');
const sendBtn       = document.getElementById('send-btn');
const chipContainer = document.getElementById('suggestion-chips');

let isProcessing = false;

// ─── Initialization ──────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  chatInput.focus();
  setupEventListeners();
});

function setupEventListeners() {
  // Send on click
  sendBtn.addEventListener('click', handleSend);

  // Send on Enter (Shift+Enter for newline)
  chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  });

  // Enable/disable send button based on input
  chatInput.addEventListener('input', () => {
    sendBtn.disabled = chatInput.value.trim() === '' || isProcessing;
    autoResizeTextarea();
  });

  // Suggestion chips
  chipContainer.addEventListener('click', (e) => {
    const chip = e.target.closest('.chip');
    if (chip) {
      const query = chip.getAttribute('data-query');
      chatInput.value = query;
      sendBtn.disabled = false;
      handleSend();
    }
  });
}

// ─── Auto-resize Textarea ────────────────────────────────────
function autoResizeTextarea() {
  chatInput.style.height = 'auto';
  chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
}

// ─── Send Message ────────────────────────────────────────────
async function handleSend() {
  const text = chatInput.value.trim();
  if (!text || isProcessing) return;

  isProcessing = true;
  sendBtn.disabled = true;

  // Hide welcome screen on first message
  if (welcomeScreen) {
    welcomeScreen.classList.add('hidden');
  }

  // Render user message
  appendMessage('user', text);

  // Clear input
  chatInput.value = '';
  chatInput.style.height = 'auto';

  // Show typing indicator
  const typingEl = showTypingIndicator();

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text })
    });

    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }

    const data = await response.json();

    // Remove typing indicator
    removeTypingIndicator(typingEl);

    // Render agent response
    appendMessage('agent', data.report, data.pdf_url);

  } catch (error) {
    removeTypingIndicator(typingEl);
    appendMessage('agent', 'Sorry, I encountered an error connecting to the server. Please make sure the server is running and try again.');
    console.error('Chat error:', error);
  } finally {
    isProcessing = false;
    sendBtn.disabled = chatInput.value.trim() === '';
    chatInput.focus();
  }
}

// ─── Render Messages ─────────────────────────────────────────
function appendMessage(role, text, pdfUrl = null) {
  const row = document.createElement('div');
  row.className = `message-row ${role}`;

  const avatar = document.createElement('div');
  avatar.className = 'msg-avatar';
  avatar.textContent = role === 'agent' ? 'A' : 'U';

  const content = document.createElement('div');
  content.className = 'msg-content';

  if (role === 'agent') {
    content.innerHTML = markdownToHtml(text);
  } else {
    content.textContent = text;
  }

  // PDF download button
  if (pdfUrl) {
    const pdfBtn = document.createElement('a');
    pdfBtn.className = 'pdf-download-btn';
    pdfBtn.href = pdfUrl;
    pdfBtn.target = '_blank';
    pdfBtn.innerHTML = `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
        <polyline points="7 10 12 15 17 10"/>
        <line x1="12" y1="15" x2="12" y2="3"/>
      </svg>
      Download PDF Report
    `;
    content.appendChild(pdfBtn);
  }

  // Timestamp
  const time = document.createElement('div');
  time.className = 'msg-time';
  time.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  content.appendChild(time);

  row.appendChild(avatar);
  row.appendChild(content);
  chatMessages.appendChild(row);

  scrollToBottom();
}

// ─── Typing Indicator ───────────────────────────────────────
function showTypingIndicator() {
  const indicator = document.createElement('div');
  indicator.className = 'typing-indicator';
  indicator.id = 'typing-indicator';
  indicator.innerHTML = `
    <div class="msg-avatar">A</div>
    <div class="typing-dots">
      <span></span>
      <span></span>
      <span></span>
    </div>
  `;
  chatMessages.appendChild(indicator);
  scrollToBottom();
  return indicator;
}

function removeTypingIndicator(el) {
  if (el && el.parentNode) {
    el.parentNode.removeChild(el);
  }
}

// ─── Markdown → HTML (lightweight) ──────────────────────────
function markdownToHtml(text) {
  if (!text) return '';

  let html = escapeHtml(text);

  // Headers
  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

  // Bold
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

  // Italic
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

  // Horizontal rule
  html = html.replace(/^---$/gm, '<hr>');

  // Unordered lists
  html = html.replace(/^[\-\*] (.+)$/gm, '<li>$1</li>');
  html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');

  // Ordered lists
  html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

  // Paragraphs (double newline)
  html = html.replace(/\n\n/g, '</p><p>');

  // Single newline → <br>
  html = html.replace(/\n/g, '<br>');

  // Wrap in paragraph
  html = '<p>' + html + '</p>';

  // Clean up empty paragraphs
  html = html.replace(/<p>\s*<\/p>/g, '');
  html = html.replace(/<p><h([123])>/g, '<h$1>');
  html = html.replace(/<\/h([123])><\/p>/g, '</h$1>');
  html = html.replace(/<p><hr><\/p>/g, '<hr>');
  html = html.replace(/<p><ul>/g, '<ul>');
  html = html.replace(/<\/ul><\/p>/g, '</ul>');

  return html;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// ─── Scroll ──────────────────────────────────────────────────
function scrollToBottom() {
  requestAnimationFrame(() => {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  });
}
