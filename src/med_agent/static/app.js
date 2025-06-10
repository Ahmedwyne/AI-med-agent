document.addEventListener('DOMContentLoaded', function() {
  const form = document.getElementById('query-form');
  const input = document.getElementById('user-query');
  const conversation = document.getElementById('conversation');

  form.addEventListener('submit', async function(e) {
    e.preventDefault();
    const userText = input.value.trim();
    if (!userText) return;
    renderMessage(userText, 'user-msg');
    input.value = '';
    try {
      const res = await fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userText })
      });
      const data = await res.json();
      if (data.result) {
        renderMessage(data.result, 'agent-msg');
      } else if (data.error) {
        emitAgentError('Error: ' + data.error);
        renderMessage('Error: ' + data.error, 'error');
      }
    } catch (err) {
      emitAgentError('Network error: ' + err);
      renderMessage('Network error: ' + err, 'error');
    }
  });

  // To enable Markdown rendering, include marked.js in your HTML:
  // <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  // The renderMessage function will use marked if available, otherwise fallback to innerHTML.

  function renderMessage(text, cls) {
    const div = document.createElement('div');
    div.className = cls;
    if (cls === 'agent-msg' && window.marked) {
      // Render agent answers as Markdown (supports headings, lists, links, etc.)
      div.innerHTML = window.marked.parse(text);
    } else {
      div.textContent = text;
    }
    conversation.appendChild(div);
    conversation.scrollTop = conversation.scrollHeight;
  }

  // Emit custom error event for feedback UI
  function emitAgentError(msg) {
    const event = new CustomEvent('agent-error', { detail: msg });
    window.dispatchEvent(event);
  }
});
