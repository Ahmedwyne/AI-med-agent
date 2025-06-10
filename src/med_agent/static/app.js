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
        renderMessage('Error: ' + data.error, 'error');
      }
    } catch (err) {
      renderMessage('Network error: ' + err, 'error');
    }
  });

  function renderMessage(text, cls) {
    const div = document.createElement('div');
    div.className = cls;
    div.innerHTML = text;
    conversation.appendChild(div);
    conversation.scrollTop = conversation.scrollHeight;
  }
});
