const chatLog = document.getElementById("chatLog");
const inputSlot = document.getElementById("inputSlot");
const chatForm = document.getElementById("chatForm");
const resultsEmpty = document.getElementById("resultsEmpty");
const resultsList = document.getElementById("resultsList");
const engineBadge = document.getElementById("engineBadge");

chatForm.addEventListener("submit", (e) => e.preventDefault());

const state = {
  skill_level: null,
  available_hardware: [],
  interests: null,
  time_budget_hours: null,
  componentCatalog: [],
};

function addBubble(text, who) {
  const div = document.createElement("div");
  div.className = `bubble ${who}`;
  div.textContent = text;
  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
  return div;
}

function clearInputSlot() {
  inputSlot.innerHTML = "";
}

async function fetchHealth() {
  try {
    const r = await fetch("/api/health");
    const data = await r.json();
    engineBadge.textContent = data.llm_available
      ? "engine: fine-tuned model"
      : "engine: grounded fallback";
    engineBadge.className = "engine-badge " + (data.llm_available ? "mode-fine_tuned_llm" : "mode-template_fallback");
  } catch {
    engineBadge.textContent = "engine: unknown";
  }
}

async function fetchComponents() {
  try {
    const r = await fetch("/api/components");
    const grouped = await r.json();
    const flat = [];
    Object.values(grouped).forEach((list) => list.forEach((c) => flat.push(c)));
    state.componentCatalog = flat;
  } catch {
    state.componentCatalog = [];
  }
}

// --- Question flow -----------------------------------------------------

function askSkillLevel() {
  addBubble(
    "First - what's your skill level with electronics / microcontrollers?",
    "assistant"
  );
  clearInputSlot();

  const row = document.createElement("div");
  row.className = "choice-row";
  [
    ["beginner", "Beginner"],
    ["intermediate", "Intermediate"],
    ["advanced", "Advanced"],
  ].forEach(([value, label]) => {
    const btn = document.createElement("button");
    btn.className = "choice-btn";
    btn.type = "button";
    btn.textContent = label;
    btn.onclick = () => {
      state.skill_level = value;
      addBubble(label, "user");
      askHardware();
    };
    row.appendChild(btn);
  });
  inputSlot.appendChild(row);
}

function askHardware() {
  addBubble(
    "What hardware do you already have? Click anything that applies, or type extra items separated by commas. Leave it empty if you have nothing yet.",
    "assistant"
  );
  clearInputSlot();

  const row = document.createElement("div");
  row.className = "choice-row";
  const selected = new Set();

  const commonIds = [
    "arduino_uno", "esp32", "raspberry_pi", "ultrasonic_hcsr04", "pir_sensor",
    "dht22_temp_humidity", "servo_motor_sg90", "dc_motor", "motor_driver_l298n",
    "relay_module", "rfid_rc522", "lcd_i2c_16x2", "breadboard",
  ];
  const catalogById = Object.fromEntries(state.componentCatalog.map((c) => [c.id, c]));

  commonIds.forEach((id) => {
    const comp = catalogById[id];
    if (!comp) return;
    const btn = document.createElement("button");
    btn.className = "choice-btn";
    btn.type = "button";
    btn.textContent = comp.name;
    btn.onclick = () => {
      if (selected.has(comp.name)) {
        selected.delete(comp.name);
        btn.classList.remove("selected");
      } else {
        selected.add(comp.name);
        btn.classList.add("selected");
      }
    };
    row.appendChild(btn);
  });
  inputSlot.appendChild(row);

  const textRow = document.createElement("div");
  textRow.className = "text-input-row";
  const input = document.createElement("input");
  input.className = "text-input";
  input.placeholder = "Other parts (comma separated, optional)";
  const btn = document.createElement("button");
  btn.className = "primary-btn";
  btn.type = "submit";
  btn.textContent = "Next";
  textRow.appendChild(input);
  textRow.appendChild(btn);
  inputSlot.appendChild(textRow);

  chatForm.onsubmit = (e) => {
    e.preventDefault();
    const extra = input.value.split(",").map((s) => s.trim()).filter(Boolean);
    const all = [...selected, ...extra];
    state.available_hardware = all;
    addBubble(all.length ? all.join(", ") : "(nothing yet)", "user");
    askInterests();
  };
}

function askInterests() {
  addBubble(
    "What kind of automation are you interested in? (e.g. robotics, safety systems, IoT monitoring, motion control) - a sentence is fine.",
    "assistant"
  );
  clearInputSlot();

  const textRow = document.createElement("div");
  textRow.className = "text-input-row";
  const input = document.createElement("input");
  input.className = "text-input";
  input.placeholder = "e.g. robotic arms and sorting systems";
  const btn = document.createElement("button");
  btn.className = "primary-btn";
  btn.type = "submit";
  btn.textContent = "Next";
  textRow.appendChild(input);
  textRow.appendChild(btn);
  inputSlot.appendChild(textRow);
  input.focus();

  chatForm.onsubmit = (e) => {
    e.preventDefault();
    state.interests = input.value.trim();
    addBubble(state.interests || "(no particular preference)", "user");
    askTimeBudget();
  };
}

function askTimeBudget() {
  addBubble("How many hours do you want to budget for this build?", "assistant");
  clearInputSlot();

  const textRow = document.createElement("div");
  textRow.className = "text-input-row";
  const input = document.createElement("input");
  input.className = "number-input";
  input.type = "number";
  input.min = "1";
  input.max = "200";
  input.value = "8";
  const btn = document.createElement("button");
  btn.className = "primary-btn";
  btn.type = "submit";
  btn.textContent = "Get suggestions";
  textRow.appendChild(input);
  textRow.appendChild(btn);
  inputSlot.appendChild(textRow);
  input.focus();

  chatForm.onsubmit = (e) => {
    e.preventDefault();
    const hours = parseFloat(input.value) || 8;
    state.time_budget_hours = hours;
    addBubble(`${hours} hours`, "user");
    runSuggest();
  };
}

// --- Results -------------------------------------------------------------

async function runSuggest() {
  clearInputSlot();
  const thinking = addBubble("Scoring projects against your hardware and skill level…", "thinking");

  try {
    const r = await fetch("/api/suggest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        skill_level: state.skill_level,
        available_hardware: state.available_hardware,
        interests: state.interests,
        time_budget_hours: state.time_budget_hours,
      }),
    });
    const data = await r.json();
    thinking.remove();
    addBubble(
      `Here are ${data.suggestions.length} projects that fit${data.notes ? " (" + data.notes + ")" : ""}.`,
      "assistant"
    );
    renderResults(data);
  } catch (err) {
    thinking.remove();
    addBubble("Something went wrong reaching the suggestion engine. Please try again.", "assistant");
  }

  addRestartButton();
}

function renderResults(data) {
  resultsEmpty.style.display = "none";
  resultsList.innerHTML = "";

  data.suggestions.forEach((s) => {
    const card = document.createElement("div");
    card.className = "card";

    const scorePct = Math.round(s.match_score * 100);

    card.innerHTML = `
      <div class="card-top">
        <h3 class="card-title">${escapeHtml(s.title)}</h3>
      </div>
      <div class="badges">
        <span class="badge skill-${s.skill_level}">${s.skill_level}</span>
        <span class="badge">${escapeHtml(s.category)}</span>
        <span class="badge">~${s.time_hours}h</span>
        <span class="badge">${scorePct}% match</span>
      </div>
      <div class="score-bar-track"><div class="score-bar-fill" style="width:${scorePct}%"></div></div>
      <p class="card-desc">${escapeHtml(s.description)}</p>
      <div class="parts-row">
        <div class="parts-col have">
          <h4>You have</h4>
          <ul>${s.have_components.map((c) => `<li>${escapeHtml(c)}</li>`).join("") || "<li>none</li>"}</ul>
        </div>
        <div class="parts-col missing">
          <h4>You need</h4>
          <ul>${s.missing_components.map((c) => `<li>${escapeHtml(c)}</li>`).join("") || "<li>none</li>"}</ul>
        </div>
      </div>
      <div class="rationale-box">${escapeHtml(s.rationale)}</div>
      <span class="source-tag">${s.source === "fine_tuned_llm" ? "personalized by fine-tuned model" : "grounded suggestion engine"}</span>
    `;
    resultsList.appendChild(card);
  });
}

function addRestartButton() {
  clearInputSlot();
  const btn = document.createElement("button");
  btn.className = "primary-btn ghost";
  btn.type = "button";
  btn.textContent = "Start over";
  btn.onclick = () => {
    state.skill_level = null;
    state.available_hardware = [];
    state.interests = null;
    state.time_budget_hours = null;
    chatLog.innerHTML = "";
    resultsList.innerHTML = "";
    resultsEmpty.style.display = "block";
    askSkillLevel();
  };
  inputSlot.appendChild(btn);
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}

// --- Init ------------------------------------------------------------------

(async function init() {
  await Promise.all([fetchHealth(), fetchComponents()]);
  addBubble(
    "Hi! I'll ask a few quick questions and suggest scoped industrial automation / mechatronics projects that fit your setup.",
    "assistant"
  );
  askSkillLevel();
})();
