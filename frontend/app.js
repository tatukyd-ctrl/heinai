// frontend/app.js
document.addEventListener('DOMContentLoaded', () => {
  const API_STREAM = "/chat/stream";
  const API_SYNC = "/chat";

  // Lấy các phần tử DOM
  const messagesEl = document.getElementById('messages');
  const inputEl = document.getElementById('input');
  const sendBtn = document.getElementById('send-btn');
  const stopBtn = document.getElementById('stop-btn');
  const templateSelect = document.getElementById('template-select');
  const toggleThemeBtn = document.getElementById('toggle-theme');
  const clearBtn = document.getElementById('clear-chat');
  const newChatBtn = document.getElementById('new-chat');
  const chatTitle = document.getElementById('chat-title');

  // Kiểm tra phần tử DOM
  const requiredElements = {
    messagesEl, inputEl, sendBtn, stopBtn, templateSelect, toggleThemeBtn, clearBtn, newChatBtn, chatTitle
  };
  for (const [key, value] of Object.entries(requiredElements)) {
    if (!value) {
      console.error(`Lỗi: Phần tử DOM '${key}' không tìm thấy. Kiểm tra ID trong index.html.`);
      return;
    }
  }

  const STORAGE_KEY = 'bot4code.conversation.v1';

  let conversation = loadConversation() || {
    id: Date.now(),
    title: "Cuộc trò chuyện mới",
    messages: [{ role: 'system', content: 'Bạn là CodeBot — hỗ trợ về lập trình.' }]
  };

  let controller = null;
  let isStreaming = false;

  // Khởi tạo theme
  if (localStorage.getItem('theme') === 'light') {
    document.body.classList.add('light');
    toggleThemeBtn.textContent = '☀️';
  } else {
    toggleThemeBtn.textContent = '🌙';
  }

  // Sự kiện đổi theme
  toggleThemeBtn.addEventListener('click', () => {
    console.log('Đổi theme được gọi');
    document.body.classList.toggle('light');
    toggleThemeBtn.textContent = document.body.classList.contains('light') ? '☀️' : '🌙';
    localStorage.setItem('theme', document.body.classList.contains('light') ? 'light' : 'dark');
  });

  // Sự kiện gửi tin nhắn
  sendBtn.addEventListener('click', () => {
    console.log('Nút Gửi được bấm');
    sendPrompt();
  });

  inputEl.addEventListener('keydown', (e) => {
    console.log('Phím được nhấn:', e.key, 'Shift:', e.shiftKey);
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      console.log('Gửi tin nhắn bằng phím Enter');
      sendPrompt();
    }
  });

  stopBtn.addEventListener('click', () => {
    console.log('Nút Dừng được bấm');
    if (controller) controller.abort();
  });

  clearBtn.addEventListener('click', () => {
    console.log('Nút Xóa được bấm');
    conversation.messages = conversation.messages.filter(m => m.role === 'system');
    saveConversation();
    renderMessages();
  });

  newChatBtn.addEventListener('click', () => {
    console.log('Nút Tạo cuộc trò chuyện mới được bấm');
    conversation = {
      id: Date.now(),
      title: 'Cuộc trò chuyện mới',
      messages: conversation.messages.filter(m => m.role === 'system')
    };
    saveConversation();
    renderMessages();
  });

  function renderMessages() {
    messagesEl.innerHTML = '';
    chatTitle.innerText = conversation.title || 'Cuộc trò chuyện mới';
    conversation.messages.forEach(m => {
      const node = document.createElement('div');
      node.className = 'message ' + (m.role === 'user' ? 'user' : 'bot');
      const bubble = document.createElement('div');
      bubble.className = 'bubble';
      const content = document.createElement('div');
      content.className = 'content';
      if (m.role === 'user') content.textContent = m.content;
      else content.innerHTML = renderMarkdown(m.content);
      bubble.appendChild(content);
      node.appendChild(bubble);
      messagesEl.appendChild(node);

      if (m.role !== 'user') {
        node.querySelectorAll('pre').forEach(pre => {
          if (pre.querySelector('.copy-btn')) return;
          const btn = document.createElement('button');
          btn.className = 'copy-btn';
          btn.textContent = 'Sao chép';
          btn.addEventListener('click', () => {
            const code = pre.querySelector('code') || pre;
            navigator.clipboard.writeText(code.textContent);
            btn.textContent = 'Đã sao chép!';
            setTimeout(() => btn.textContent = 'Sao chép', 1200);
          });
          pre.appendChild(btn);
        });
      }
    });
    messagesEl.scrollTop = messagesEl.scrollHeight;
    if (typeof Prism !== 'undefined') {
      Prism.highlightAll();
    } else {
      console.warn('Prism.js không tải được, bỏ qua tô màu mã nguồn');
    }
  }

  function escapeHtml(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function renderMarkdown(text) {
    let html = text.replace(/```(\w+)?\n([\s\S]*?)```/g, (m, lang, code) => {
      const cls = lang ? `language-${lang}` : '';
      return `<pre><code class="${cls}">${escapeHtml(code)}</code></pre>`;
    });
    return html.split(/(<pre>[\s\S]*?<\/pre>)/g).map(chunk => {
      if (chunk.startsWith('<pre>')) return chunk;
      return escapeHtml(chunk).replace(/\n/g, '<br>');
    }).join('');
  }

  function saveConversation() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(conversation));
    } catch (e) {
      console.error('Lỗi lưu cuộc trò chuyện:', e);
    }
  }

  function loadConversation() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      return raw ? JSON.parse(raw) : null;
    } catch (e) {
      console.error('Lỗi tải cuộc trò chuyện:', e);
      return null;
    }
  }

  async function sendPrompt() {
    if (isStreaming) {
      console.warn('Đang stream, không thể gửi thêm tin nhắn');
      return;
    }
    const prompt = inputEl.value.trim();
    if (!prompt) {
      console.warn('Không gửi: Tin nhắn trống');
      return;
    }

    conversation.messages.push({ role: 'user', content: prompt });
    if (!conversation.title || conversation.title === 'Cuộc trò chuyện mới') {
      conversation.title = prompt.split('\n')[0].slice(0, 80);
    }
    inputEl.value = '';
    saveConversation();
    renderMessages();

    const assistantMsg = { role: 'assistant', content: '' };
    conversation.messages.push(assistantMsg);
    saveConversation();
    renderMessages();

    isStreaming = true;
    controller = new AbortController();
    setLoading(true);

    try {
      const payload = { messages: conversation.messages, template: templateSelect.value, provider: 'auto' };
      console.log('Gửi yêu cầu tới', API_STREAM, 'với payload:', payload);
      const resp = await fetch(API_STREAM, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: controller.signal
      });

      if (!resp.ok) {
        const data = await resp.json().catch(() => ({ error: resp.statusText }));
        console.error('Lỗi API:', data);
        assistantMsg.content = data.reply || `Lỗi: ${data.error || resp.status}`;
        saveConversation();
        renderMessages();
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
          const markerIdx = buffer.indexOf(FILE_MARKER);
          if (markerIdx !== -1) {
            const before = buffer.slice(0, markerIdx);
            assistantMsg.content += before;
            const after = buffer.slice(markerIdx + FILE_MARKER.length).trim();
            if (after) assistantMsg.content += `.`;
            buffer = "";
          } else {
            assistantMsg.content += buffer;
            buffer = "";
          }
          saveConversation();
          renderMessages();
        }
      }
      if (buffer) {
        assistantMsg.content += buffer;
        saveConversation();
        renderMessages();
      }
    } catch (err) {
      console.error('Lỗi fetch:', err);
      if (err.name === 'AbortError') {
        assistantMsg.content += "\n\n[Đã dừng stream bởi người dùng]";
      } else {
        assistantMsg.content += `\n\n[Lỗi mạng] ${err.message}`;
      }
      saveConversation();
      renderMessages();
    } finally {
      isStreaming = false;
      controller = null;
      setLoading(false);
    }
  }

  function setLoading(isLoading) {
    sendBtn.disabled = isLoading;
    stopBtn.disabled = !isLoading;
    sendBtn.textContent = isLoading ? 'Đang tải...' : 'Gửi';
  }

  renderMessages();
});
