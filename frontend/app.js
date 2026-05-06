const state = {
  sessionId: null,
  status: "idle",
  guide: "",
};

const elements = {
  planForm: document.getElementById("plan-form"),
  requestInput: document.getElementById("request-input"),
  resumeSection: document.getElementById("resume-section"),
  resumeForm: document.getElementById("resume-form"),
  resumeInput: document.getElementById("resume-input"),
  questionText: document.getElementById("question-text"),
  feedbackSection: document.getElementById("feedback-section"),
  feedbackForm: document.getElementById("feedback-form"),
  feedbackInput: document.getElementById("feedback-input"),
  sessionId: document.getElementById("session-id"),
  statusText: document.getElementById("status-text"),
  flashMessage: document.getElementById("flash-message"),
  emptyState: document.getElementById("empty-state"),
  guideContent: document.getElementById("guide-content"),
  buttons: document.querySelectorAll("button"),
};

elements.planForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  await submitRequest("/plan", { request: elements.requestInput.value.trim() });
});

elements.resumeForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.sessionId) {
    showFlash("当前还没有可继续的 session。", "error");
    return;
  }
  await submitRequest(`/plan/${state.sessionId}/resume`, {
    reply: elements.resumeInput.value.trim(),
  });
});

elements.feedbackForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.sessionId) {
    showFlash("当前还没有可反馈的攻略。", "error");
    return;
  }
  await submitRequest(`/plan/${state.sessionId}/feedback`, {
    feedback: elements.feedbackInput.value.trim(),
  });
});

async function submitRequest(url, payload) {
  if (Object.values(payload).some((value) => !value)) {
    showFlash("请先填写完整内容。", "error");
    return;
  }

  setBusy(true);
  showFlash("正在和后端交互，请稍候。", "info");

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "请求失败");
    }

    applyResponse(data);
  } catch (error) {
    showFlash(error.message || "请求失败", "error");
  } finally {
    setBusy(false);
  }
}

function applyResponse(data) {
  state.sessionId = data.session_id;
  state.status = data.status;
  state.guide = data.guide || "";

  elements.sessionId.textContent = state.sessionId;
  elements.statusText.textContent =
    state.status === "completed" ? "completed" : "needs_input";

  if (data.status === "needs_input") {
    // 当图被 interrupt 时，页面切换到“补充信息”模式。
    elements.questionText.textContent = data.question || "请补充缺失信息。";
    elements.resumeSection.classList.remove("is-hidden");
    elements.feedbackSection.classList.add("is-hidden");
    elements.guideContent.classList.add("is-hidden");
    elements.emptyState.classList.remove("is-hidden");
    elements.emptyState.querySelector(".empty-title").textContent = "需要补充信息";
    elements.emptyState.querySelector(".empty-copy").textContent =
      "左侧已经展示后端返回的问题。补充后，LangGraph 会从中断点继续执行。";
    showFlash("后端需要更多信息后才能继续规划。", "info");
    return;
  }

  // completed 时渲染攻略，并开放反馈入口。
  elements.resumeSection.classList.add("is-hidden");
  elements.feedbackSection.classList.remove("is-hidden");
  elements.emptyState.classList.add("is-hidden");
  elements.guideContent.classList.remove("is-hidden");
  elements.guideContent.innerHTML = renderMarkdown(data.guide || "");
  elements.resumeInput.value = "";
  showFlash("攻略已生成，你可以继续在下方提交局部反馈。", "info");
}

function setBusy(isBusy) {
  elements.buttons.forEach((button) => {
    button.disabled = isBusy;
  });
}

function showFlash(message, level) {
  if (!message) {
    elements.flashMessage.className = "flash is-hidden";
    elements.flashMessage.textContent = "";
    return;
  }

  elements.flashMessage.className = `flash ${level}`;
  elements.flashMessage.textContent = message;
}

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function renderInline(text) {
  return escapeHtml(text).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
}

function renderMarkdown(markdown) {
  // 这里不引第三方 markdown 库，只实现这个项目实际会返回的结构：
  // 标题、段落、无序列表。这样更便于学习，也不会平白增加依赖。
  const lines = markdown.split("\n");
  const parts = [];
  let listBuffer = [];

  const flushList = () => {
    if (!listBuffer.length) {
      return;
    }
    parts.push(`<ul>${listBuffer.join("")}</ul>`);
    listBuffer = [];
  };

  for (const rawLine of lines) {
    const line = rawLine.trim();

    if (!line) {
      flushList();
      continue;
    }

    if (line.startsWith("- ")) {
      listBuffer.push(`<li>${renderInline(line.slice(2))}</li>`);
      continue;
    }

    flushList();

    if (line.startsWith("### ")) {
      parts.push(`<h3>${renderInline(line.slice(4))}</h3>`);
      continue;
    }
    if (line.startsWith("## ")) {
      parts.push(`<h2>${renderInline(line.slice(3))}</h2>`);
      continue;
    }
    if (line.startsWith("# ")) {
      parts.push(`<h1>${renderInline(line.slice(2))}</h1>`);
      continue;
    }

    parts.push(`<p>${renderInline(line)}</p>`);
  }

  flushList();
  return parts.join("");
}

