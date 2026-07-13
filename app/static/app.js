(() => {
  "use strict";

  const api = "/api/v1";
  const storageKey = "maple-ember-session";
  const elements = {
    form: document.querySelector("#chat-form"),
    input: document.querySelector("#chat-input"),
    send: document.querySelector("#send-button"),
    list: document.querySelector("#message-list"),
    notice: document.querySelector("#notice"),
    turns: document.querySelector("#turn-count"),
    expiry: document.querySelector("#expiry-count"),
    meta: document.querySelector(".session-meta"),
    characterCount: document.querySelector("#character-count"),
    servicePill: document.querySelector("#service-pill"),
    serviceLabel: document.querySelector("#service-label"),
    sessionAction: document.querySelector("#session-action"),
    sessionActionCopy: document.querySelector("#session-action-copy"),
    startSession: document.querySelector("#start-session"),
    newConversation: document.querySelector("#new-conversation"),
  };

  const state = {
    sessionId: null,
    expiresAt: null,
    turnsRemaining: 20,
    busy: false,
    ended: false,
    pending: null,
  };

  function loadSession() {
    try {
      const saved = JSON.parse(localStorage.getItem(storageKey));
      if (saved?.sessionId) {
        state.sessionId = saved.sessionId;
        state.expiresAt = saved.expiresAt;
        state.turnsRemaining = Number.isInteger(saved.turnsRemaining) ? saved.turnsRemaining : 20;
      }
    } catch {
      localStorage.removeItem(storageKey);
    }
  }

  function saveSession() {
    if (!state.sessionId) return;
    localStorage.setItem(storageKey, JSON.stringify({
      sessionId: state.sessionId,
      expiresAt: state.expiresAt,
      turnsRemaining: state.turnsRemaining,
    }));
  }

  function clearSession() {
    state.sessionId = null;
    state.expiresAt = null;
    state.turnsRemaining = 20;
    localStorage.removeItem(storageKey);
  }

  function newRequestId() {
    if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
    return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (token) => {
      const random = Math.floor(Math.random() * 16);
      const value = token === "x" ? random : (random & 0x3) | 0x8;
      return value.toString(16);
    });
  }

  function addMessage(text, role = "assistant") {
    const message = document.createElement("div");
    message.className = `message ${role}`;
    message.textContent = text;
    elements.list.append(message);
    elements.list.scrollTop = elements.list.scrollHeight;
    return message;
  }

  function addTyping() {
    const message = document.createElement("div");
    message.className = "message pending";
    message.setAttribute("aria-label", "Qwen is preparing a reply");
    for (let index = 0; index < 3; index += 1) {
      const dot = document.createElement("span");
      dot.className = "typing-dot";
      message.append(dot);
    }
    elements.list.append(message);
    elements.list.scrollTop = elements.list.scrollHeight;
    return message;
  }

  function showWelcome() {
    addMessage("Good evening. I can help you explore the menu, find dishes for dietary preferences, check our hours, or plan a reservation.");
  }

  function showNotice(message) {
    elements.notice.textContent = message;
    elements.notice.hidden = !message;
  }

  function setBusy(busy) {
    state.busy = busy;
    elements.input.disabled = busy || state.ended;
    elements.send.disabled = busy || state.ended;
    document.querySelectorAll(".prompt-chip").forEach((button) => { button.disabled = busy || state.ended; });
  }

  function endSession(message) {
    state.ended = true;
    state.pending = null;
    clearSession();
    elements.sessionActionCopy.textContent = message;
    elements.sessionAction.hidden = false;
    setBusy(false);
  }

  function updateSessionMeta() {
    elements.turns.textContent = `${state.turnsRemaining} turn${state.turnsRemaining === 1 ? "" : "s"}`;
    let minutes = 30;
    if (state.expiresAt) {
      minutes = Math.max(0, Math.ceil((new Date(state.expiresAt).getTime() - Date.now()) / 60000));
    }
    elements.expiry.textContent = minutes > 0 ? `${minutes} min` : "expired";
    const warning = state.turnsRemaining <= 3 || minutes <= 5;
    elements.meta.classList.toggle("warning", warning);
    if (state.expiresAt && minutes <= 0 && !state.ended && !state.busy) {
      endSession("This conversation expired after being idle. Start a new one when you're ready.");
    }
  }

  async function createSession() {
    const response = await fetch(`${api}/sessions`, { method: "POST" });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.detail || "The concierge is busy. Please try again shortly.");
    state.sessionId = body.session_id;
    state.expiresAt = body.expires_at;
    state.turnsRemaining = body.turns_remaining;
    state.ended = false;
    saveSession();
    updateSessionMeta();
  }

  function errorMessage(status, detail) {
    if (status === 404) return "This conversation is no longer available, likely because the server restarted.";
    if (status === 410) return "This conversation has expired.";
    if (status === 409) return "That request conflicted with an earlier retry. Please send it again as a new message.";
    if (status === 422) return detail || "Please check the message and try again.";
    if (status === 429 && /turn limit/i.test(detail || "")) return "You have reached this conversation's turn limit.";
    if (status === 429) return "The restaurant chat is at capacity. Please try again shortly.";
    if (status === 503) return "Qwen is warming up or temporarily unavailable. Your message is ready to retry.";
    return detail || "Something went wrong while preparing the reply.";
  }

  function addRetry(message) {
    const wrapper = document.createElement("div");
    wrapper.className = "message system";
    const label = document.createElement("span");
    label.textContent = `${message} `;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "text-button";
    button.textContent = "Retry";
    button.addEventListener("click", () => {
      wrapper.remove();
      performRequest(state.pending);
    }, { once: true });
    wrapper.append(label, button);
    elements.list.append(wrapper);
    elements.list.scrollTop = elements.list.scrollHeight;
  }

  async function performRequest(request) {
    if (!request || state.busy || state.ended) return;
    setBusy(true);
    showNotice("");
    const typing = addTyping();
    try {
      const response = await fetch(`${api}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: request.message,
          session_id: state.sessionId,
          request_id: request.requestId,
        }),
      });
      const body = await response.json().catch(() => ({}));
      typing.remove();
      if (!response.ok) {
        const message = errorMessage(response.status, body.detail);
        if ([404, 410].includes(response.status) || (response.status === 429 && /turn limit/i.test(body.detail || ""))) {
          addMessage(message, "system");
          endSession(`${message} Start a new conversation to continue.`);
        } else if ([409, 422].includes(response.status)) {
          addMessage(message, "system");
          state.pending = null;
        } else {
          addRetry(message);
        }
        return;
      }
      addMessage(body.response);
      state.sessionId = body.session_id;
      state.expiresAt = body.expires_at;
      state.turnsRemaining = body.turns_remaining;
      state.pending = null;
      saveSession();
      updateSessionMeta();
      if (body.limit_warning) showNotice(body.limit_warning);
      if (body.turns_remaining === 0) endSession("This conversation has reached its turn limit. Start a new one to continue.");
    } catch {
      typing.remove();
      addRetry("The connection was interrupted. Your message is ready to retry.");
    } finally {
      setBusy(false);
      if (!state.ended) elements.input.focus();
    }
  }

  async function submitMessage(rawMessage) {
    const message = rawMessage.trim();
    if (!message || state.busy || state.ended) return;
    if (!state.sessionId) {
      try {
        setBusy(true);
        await createSession();
      } catch (error) {
        showNotice(error.message);
        setBusy(false);
        return;
      }
    }
    addMessage(message, "user");
    elements.input.value = "";
    resizeInput();
    updateCharacterCount();
    state.pending = { message, requestId: newRequestId() };
    setBusy(false);
    await performRequest(state.pending);
  }

  async function startNewConversation() {
    const previous = state.sessionId;
    if (previous) fetch(`${api}/sessions/${previous}`, { method: "DELETE" }).catch(() => {});
    clearSession();
    state.ended = false;
    state.pending = null;
    elements.sessionAction.hidden = true;
    showNotice("");
    elements.list.replaceChildren();
    showWelcome();
    updateSessionMeta();
    setBusy(false);
    elements.input.focus();
  }

  function resizeInput() {
    elements.input.style.height = "auto";
    elements.input.style.height = `${Math.min(elements.input.scrollHeight, 120)}px`;
  }

  function updateCharacterCount() {
    elements.characterCount.textContent = `${elements.input.value.length} / 2000`;
  }

  async function checkService() {
    try {
      const response = await fetch("/ready");
      const body = await response.json();
      if (!response.ok) throw new Error();
      elements.servicePill.dataset.state = "online";
      elements.serviceLabel.textContent = body.model ? `${body.model} ready` : "Concierge ready";
    } catch {
      elements.servicePill.dataset.state = "offline";
      elements.serviceLabel.textContent = "Concierge unavailable";
    }
  }

  elements.form.addEventListener("submit", (event) => {
    event.preventDefault();
    submitMessage(elements.input.value);
  });
  elements.input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      elements.form.requestSubmit();
    }
  });
  elements.input.addEventListener("input", () => { resizeInput(); updateCharacterCount(); });
  document.querySelectorAll(".prompt-chip").forEach((button) => {
    button.addEventListener("click", () => submitMessage(button.dataset.prompt));
  });
  elements.newConversation.addEventListener("click", startNewConversation);
  elements.startSession.addEventListener("click", startNewConversation);

  loadSession();
  showWelcome();
  updateSessionMeta();
  setInterval(updateSessionMeta, 15000);
  checkService();
})();
