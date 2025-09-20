// frontend/app.js
// Robust streaming client + localStorage + copy code + Prism highlight

const API_STREAM = "/chat/stream"; // Changed to relative URL
const API_SYNC = "/chat"; // Changed to relative URL

const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send-btn');
const stopBtn = document.getElementById('stop-btn');
const templateSelect = document.getElementById('template-select');
const toggleThemeBtn = document.getElementById('toggle-theme');
const clearBtn = document.getElementById('clear-chat');
const newChatBtn = document.getElementById('new-chat');
const chatTitle = document.getElementById('chat-title');

const STORAGE_KEY = 'bot4code.conversation.v1';

// load or init conversation
let conversation = loadConversation() || {
  id: Date.now(),
  title: "New chat",
  messages: [ { role: 'system', content: 'You are CodeBot — help with code.' } ]
};

let controller = null;
let isStreaming = false;

// theme
if (localStorage.getItem('theme') === 'light') {
  document.body.classList.add('light');
  toggleThemeBtn.textContent = '☀️';
} else {
  toggleThemeBtn.textContent = '🌙';
}
toggleThemeBtn.addEventListener('click', () => {
  document.body.classList.toggle('light');
  toggleThemeBtn.textContent = document.body.classList.contains('light') ? '☀️' : '🌙';
  localStorage.setItem('theme', document.body.classList.contains('light') ? 'light' : 'dark');
});

// handlers
sendBtn.addEventListener('click', sendPrompt);
inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendPrompt(); }
});
stopBtn.addEventListener('click', () => { if (controller) controller.abort(); });
clearBtn.addEventListener('click', () => { 
  conversation.messages = conversation.messages.filter(m => m.role === 'system'); 
  saveConversation(); 
  renderMessages(); 
});
newChatBtn.addEventListener('click', () => { 
  conversation = { id: Date.now(), title: 'New chat', messages: conversation.messages.filter(m => m.role === 'system') }; 
  saveConversation(); 
  renderMessages(); 
});

// render
function renderMessages() {
  messagesEl.innerHTML = '';
  chatTitle.innerText = conversation.title || 'New chat';
  conversation.messages.forEach(m => {
    const node = document.createElement('div');
    node.className = 'message ' + (m.role === 'user' ? 'user' : 'bot');
    const bubble = document.createElement('div'); bubble.className = 'bubble';
    const content = document.createElement('div'); content.className = 'content';
    if (m.role === 'user') content.textContent = m.content;
    else content.innerHTML = renderMarkdown(m.content);
    bubble.appendChild(content); node.appendChild(bubble);
    messagesEl.appendChild(node);

    // add copy button to code blocks
    if (m.role !== 'user') {
      node.querySelectorAll('pre').forEach(pre => {
        if (pre.querySelector('.copy-btn')) return;
        const btn = document.createElement('button'); btn.className = 'copy-btn'; btn.textContent = 'Copy';
        btn.addEventListener('click', () => {
          const code = pre.querySelector('code') || pre;
          navigator.clipboard.writeText(code.textContent);
          btn.textContent = 'Copied!';
          setTimeout(() => btn.textContent = 'Copy', 1200);
        });
        pre.appendChild(btn);
      });
    }
  });
  messagesEl.scrollTop = messagesEl.scrollHeight;
  Prism.highlightAll();
}

function escapeHtml(s) { 
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); 
}
function renderMarkdown(text) {
  let html = text.replace(/```(\w+)?\n([\s\S]*?)```/g, (m, lang, code) => {
    const cls = lang ? `language-${lang}` : '';
    return `<pre><code class="${cls}">${escapeHtml(code)}</code></pre>`;
  });
  // preserve newlines outside code
  return html.split(/(<pre>[\s\S]*?<\/pre>)/g).map(chunk => {
    if (chunk.startsWith('<pre>')) return chunk;
    return escapeHtml(chunk).replace(/\n/g, '<br>');
  }).join('');
}

function saveConversation() { 
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(conversation)); } catch(e) {} 
}
function loadConversation() { 
  try { const raw = localStorage.getItem(STORAGE_KEY); return raw ? JSON.parse(raw) : null; } catch(e) { return null; } 
}

// streaming send
async function sendPrompt() {
  if (isStreaming) return;
  const prompt = inputEl.value.trim();
  if (!prompt) return;

  // push user message
  conversation.messages.push({ role: 'user', content: prompt });
  if (!conversation.title || conversation.title === 'New chat') conversation.title = prompt.split('\n')[0].slice(0, 80);
  inputEl.value = '';
  saveConversation();
  renderMessages();

  // add assistant placeholder
  const assistantMsg = { role: 'assistant', content: '' };
  conversation.messages.push(assistantMsg);
  saveConversation();
  renderMessages();

  isStreaming = true;
  controller = new AbortController();
  setLoading(true);

  try {
    const payload = { messages: conversation.messages, template: templateSelect.value, provider: 'auto' };
    const resp = await fetch(API_STREAM, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: controller.signal
    });

    if (!resp.ok) {
      // fallback to sync endpoint or show error
      const data = await resp.json().catch(() => ({ error: resp.statusText }));
      assistantMsg.content = data.reply || `Error: ${data.error || resp.status}`;
      saveConversation(); renderMessages();
      return;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    const FILE_MARKER = "[FILE_UPLOADED]";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      if (value) {
        const chunk = decoder.decode(value, { stream: true });
        buffer += chunk;
        // process marker robustly (marker may be split across chunks)
        const markerIdx = buffer.indexOf(FILE_MARKER);
        if (markerIdx !== -1) {
          // everything before marker is text; after marker contains link (maybe incomplete)
          const before = buffer.slice(0, markerIdx);
          assistantMsg.content += before;
          const after = buffer.slice(markerIdx + FILE_MARKER.length).trim();
          // append uploaded link note (if available)
          if (after) assistantMsg.content += `.`;
          buffer = "";
        } else {
          // no marker yet -> append all to assistant
          assistantMsg.content += buffer;
          buffer = "";
        }
        saveConversation();
        renderMessages();
      }
    }
    // if any leftover buffer, append
    if (buffer) {
      assistantMsg.content += buffer;
      saveConversation(); renderMessages();
    }
  } catch (err) {
    if (err.name === 'AbortError') {
      assistantMsg.content += "\n\n[Stream stopped by user]";
    } else {
      assistantMsg.content += `\n\n[Network error] ${err.message}`;
    }
    saveConversation(); renderMessages();
  } finally {
    isStreaming = false;
    controller = null;
    setLoading(false);
  }
}

function setLoading(isLoading) {
  sendBtn.disabled = isLoading;
  stopBtn.disabled = !isLoading;
  sendBtn.textContent = isLoading ? 'Loading...' : 'Send';
}

// initial render
renderMessages();
