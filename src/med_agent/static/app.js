document.addEventListener('DOMContentLoaded', function () {
  const form       = document.getElementById('query-form');
  const textarea   = document.getElementById('user-query');
  const chatArea   = document.getElementById('chat-area');
  const sendBtn    = document.getElementById('send-btn');
  const menuBtn    = document.getElementById('menu-btn');
  const sidebar    = document.querySelector('.sidebar');
  const toast      = document.getElementById('toast');

  // ── Sidebar toggle ──
  menuBtn.addEventListener('click', () => sidebar.classList.toggle('collapsed'));

  // ── Example query buttons ──
  document.querySelectorAll('.example-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      textarea.value = btn.dataset.query;
      autoResize();
      textarea.focus();
      if (window.innerWidth <= 640) sidebar.classList.add('collapsed');
    });
  });

  // ── Auto-resize textarea ──
  function autoResize() {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 160) + 'px';
  }
  textarea.addEventListener('input', autoResize);

  // ── Send on Enter, newline on Shift+Enter ──
  textarea.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      form.requestSubmit();
    }
  });

  // ── Submit ──
  form.addEventListener('submit', async function (e) {
    e.preventDefault();
    const query = textarea.value.trim();
    if (!query) return;

    hideWelcome();
    appendMessage(query, 'user');
    textarea.value = '';
    textarea.style.height = 'auto';

    sendBtn.disabled = true;
    const typingEl = appendTyping();

    try {
      const res = await fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
      });
      const data = await res.json();
      removeTyping(typingEl);

      if (data.result) {
        appendMessage(data.result, 'agent');
      } else if (data.error) {
        appendMessage(data.error, 'agent', true);
        showToast('Agent returned an error', 'error');
      }
    } catch (err) {
      removeTyping(typingEl);
      appendMessage('Network error — please check your connection.', 'agent', true);
      showToast('Network error', 'error');
    } finally {
      sendBtn.disabled = false;
      textarea.focus();
    }
  });

  // ── Helpers ──

  function hideWelcome() {
    const w = document.getElementById('welcome-state');
    if (w) w.remove();
  }

  function appendMessage(text, role, isError = false) {
    const row = document.createElement('div');
    row.className = `msg-row ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = role === 'user' ? 'You' : 'AI';

    const bubble = document.createElement('div');
    bubble.className = 'bubble' + (isError ? ' error' : '');

    if (role === 'agent' && !isError && window.marked) {
      marked.setOptions({ breaks: true, gfm: true });
      bubble.innerHTML = marked.parse(text);
      // Open links in new tab
      bubble.querySelectorAll('a').forEach(a => a.setAttribute('target', '_blank'));
    } else {
      bubble.textContent = text;
    }

    row.appendChild(avatar);
    row.appendChild(bubble);
    chatArea.appendChild(row);
    scrollBottom();
    return row;
  }

  function appendTyping() {
    const row = document.createElement('div');
    row.className = 'msg-row agent typing-row';

    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = 'AI';

    const bubble = document.createElement('div');
    bubble.className = 'typing-bubble';
    bubble.innerHTML = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';

    row.appendChild(avatar);
    row.appendChild(bubble);
    chatArea.appendChild(row);
    scrollBottom();
    return row;
  }

  function removeTyping(el) {
    if (el && el.parentNode) el.parentNode.removeChild(el);
  }

  function scrollBottom() {
    chatArea.scrollTo({ top: chatArea.scrollHeight, behavior: 'smooth' });
  }

  function showToast(msg, type = '') {
    toast.textContent = msg;
    toast.className = 'toast show' + (type ? ' ' + type : '');
    clearTimeout(toast._timer);
    toast._timer = setTimeout(() => { toast.className = 'toast'; }, 3500);
  }
});
