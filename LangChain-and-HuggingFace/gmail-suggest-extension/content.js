// debounce to avoid thrashing the UI
let lastCompose = null;

function getThreadText() {
  // Gmail renders each email in the thread as div.a3s
  const blocks = document.querySelectorAll("div.a3s");
  if (!blocks.length) return "";
  // join the innerText of all but the last (which is the draft) for context
  return Array.from(blocks)
    .slice(0, -1)
    .map(b => b.innerText)
    .join("\n\n");
}

function getComposeBox() {
  // the editable DIV Gmail uses for compose:
  return document.querySelector('div[aria-label="Message Body"]');
}

function injectButton() {
  const compose = getComposeBox();
  if (!compose || compose === lastCompose) return;
  lastCompose = compose;

  // make sure we only ever add one button
  if (document.getElementById("suggest-btn")) return;

  const btn = document.createElement("button");
  btn.id = "suggest-btn";
  btn.textContent = "Suggest Reply";
  btn.style = `
    margin: 8px 0;
    padding: 6px 12px;
    background: #1a73e8;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
  `;

  // insert the button right above the compose box
  compose.parentNode.insertBefore(btn, compose);

  btn.onclick = async () => {
    btn.disabled = true;
    btn.textContent = "Thinking…";

    const thread = getThreadText();
    try {
      const resp = await fetch("http://localhost:8000/suggest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ thread }),
      });
      const data = await resp.json();
      // put the suggestion into the compose editor
      compose.innerText = data.suggestion;
      btn.textContent = "Inserted!";
    } catch (e) {
      console.error(e);
      btn.textContent = "Error!";
    }
    setTimeout(() => {
      btn.disabled = false;
      btn.textContent = "Suggest Reply";
    }, 3000);
  };
}

// watch for Gmail’s Compose box to appear
const obs = new MutationObserver(injectButton);
obs.observe(document.body, { childList: true, subtree: true });
