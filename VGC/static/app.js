const replayState = {
  data: null,
  currentTurn: 1,
  autoplayTimer: null,
  frameMode: false,
};

const el = {
  tabs: [...document.querySelectorAll(".top-tab")],
  panels: [...document.querySelectorAll(".tab-panel")],
  landingCard: document.getElementById("landing-card"),
  battleScreen: document.getElementById("battle-screen"),
  turnTitle: document.getElementById("turn-title"),
  battleSubtitle: document.getElementById("battle-subtitle"),
  metaOutcome: document.getElementById("meta-outcome"),
  metaTotalReward: document.getElementById("meta-total-reward"),
  playerName: document.getElementById("player-name"),
  playerStatus: document.getElementById("player-status"),
  playerTransition: document.getElementById("player-transition"),
  playerSprite: document.getElementById("player-sprite"),
  playerHpBar: document.querySelector("#player-hp-bar span"),
  playerHpLabel: document.getElementById("player-hp-label"),
  opponentName: document.getElementById("opponent-name"),
  opponentStatus: document.getElementById("opponent-status"),
  opponentTransition: document.getElementById("opponent-transition"),
  opponentSprite: document.getElementById("opponent-sprite"),
  opponentHpBar: document.querySelector("#opponent-hp-bar span"),
  opponentHpLabel: document.getElementById("opponent-hp-label"),
  modelAction: document.getElementById("model-action"),
  opponentAction: document.getElementById("opponent-action"),
  rewardLine: document.getElementById("reward-line"),
  commentaryList: document.getElementById("commentary-list"),
  playerTeam: document.getElementById("player-team"),
  opponentTeam: document.getElementById("opponent-team"),
  validActions: document.getElementById("valid-actions"),
  summaryLines: document.getElementById("summary-lines"),
  speed: document.getElementById("speed"),
  speedValue: document.getElementById("speed-value"),
  startBtn: document.getElementById("start-btn"),
  frameBtn: document.getElementById("frame-btn"),
  prevBtn: document.getElementById("prev-btn"),
  nextBtn: document.getElementById("next-btn"),
  jumpBtn: document.getElementById("jump-btn"),
  turnInput: document.getElementById("turn-input"),
};

function toShowdownName(name) {
  return name.toLowerCase().replace(/[^a-z0-9]/g, "");
}

function spriteCandidates(name) {
  const s = toShowdownName(name);
  return [
    `/static/sprites/${s}.png`,
    `https://play.pokemonshowdown.com/sprites/gen5/${s}.png`,
    `https://play.pokemonshowdown.com/sprites/gen4/${s}.png`,
    `https://play.pokemonshowdown.com/sprites/gen3/${s}.png`,
    `https://play.pokemonshowdown.com/sprites/gen2/${s}.png`,
  ];
}

function applySprite(img, name, mirrored) {
  const sources = spriteCandidates(name);
  img.classList.toggle("player-sprite", mirrored);
  let index = 0;
  img.onerror = () => {
    index += 1;
    if (index < sources.length) {
      img.src = sources[index];
    } else {
      img.onerror = null;
      img.alt = name;
      img.src =
        "data:image/svg+xml;utf8," +
        encodeURIComponent(
          `<svg xmlns='http://www.w3.org/2000/svg' width='180' height='180'>
            <rect width='100%' height='100%' rx='28' fill='#f7f2ef' stroke='#ff7aa2' stroke-width='3'/>
            <text x='50%' y='50%' dominant-baseline='middle' text-anchor='middle'
                  font-family='Space Grotesk' font-size='18' fill='#221830'>${name}</text>
          </svg>`
        );
    }
  };
  img.src = sources[0];
}

function hpColor(value) {
  if (value <= 20) return "#ff6b7a";
  if (value <= 50) return "#ffd166";
  return "#6de29c";
}

function setHp(target, label, hp) {
  const value = Number.isFinite(hp) ? Math.max(0, Math.min(100, hp)) : 0;
  target.style.width = `${value}%`;
  target.style.background = `linear-gradient(90deg, ${hpColor(value)} 0%, ${value <= 20 ? "#ff8793" : value <= 50 ? "#ffe29b" : "#9af0b5"} 100%)`;
  label.textContent = `${Math.round(value)}%`;
}

function actionText(label, action) {
  if (!action) return `${label}: unknown`;
  const type = action.action || action.type || "?";
  return `${label}: ${type} - ${action.choice || "?"}`;
}

function renderChips(container, items, activeName = null, hidden = 0, formatter = (x) => x) {
  container.innerHTML = "";
  items.forEach((item) => {
    const value = formatter(item);
    const chip = document.createElement("span");
    chip.className = "chip";
    if (typeof item === "string" && item === activeName) chip.classList.add("active");
    if (item?.action) chip.classList.add(item.action);
    chip.textContent = value;
    container.appendChild(chip);
  });
  for (let i = 0; i < hidden; i += 1) {
    const chip = document.createElement("span");
    chip.className = "chip hidden";
    chip.textContent = "Unknown";
    container.appendChild(chip);
  }
}

function revealedOpponent(turnNumber) {
  const revealed = [];
  replayState.data.turns.slice(0, turnNumber).forEach((turn) => {
    [turn.opponent_active_before.name, turn.opponent_active_after.name].forEach((name) => {
      if (name && name !== "unknown" && !revealed.includes(name)) revealed.push(name);
    });
  });
  return revealed;
}

function renderSummary() {
  const { meta, teams } = replayState.data;
  const playerTeam = teams.player.map((m) => m.name).join(", ");
  const opponentTeam = teams.opponent.map((m) => m.name).join(", ");
  const lines = [
    ["Outcome", String(meta.outcome || "unknown").toUpperCase()],
    ["Total Turns", String(meta.total_turns || 0)],
    ["Total Reward", Number(meta.total_reward || 0).toFixed(2)],
    ["Player Team", playerTeam],
    ["Opponent Team", opponentTeam],
    ["Model", meta.model || "Unknown"],
  ];
  el.summaryLines.innerHTML = lines
    .map(
      ([label, value]) =>
        `<div class="summary-line"><span class="label">${label}:</span><span class="value">${value}</span></div>`
    )
    .join("");
}

function renderTurn(turnIndex) {
  if (!replayState.data) return;
  const turns = replayState.data.turns;
  const turn = turns[Math.max(0, Math.min(turns.length - 1, turnIndex - 1))];
  replayState.currentTurn = turn.turn;
  el.turnInput.value = String(turn.turn);

  const pBefore = turn.player_active_before;
  const pAfter = turn.player_active_after?.name !== "unknown" ? turn.player_active_after : pBefore;
  const oBefore = turn.opponent_active_before;
  const oAfter = turn.opponent_active_after?.name !== "unknown" ? turn.opponent_active_after : oBefore;

  el.landingCard.classList.add("is-hidden");
  el.battleScreen.classList.remove("is-hidden");

  el.turnTitle.textContent = `Turn ${turn.turn} Replay`;
  el.battleSubtitle.textContent = replayState.frameMode
    ? "Frame-by-frame inspection mode"
    : "Autoplay battle replay";
  el.metaOutcome.textContent = String(replayState.data.meta.outcome || "unknown").toUpperCase();
  el.metaTotalReward.textContent = Number(replayState.data.meta.total_reward || 0).toFixed(2);

  el.playerName.textContent = pAfter.name;
  el.playerStatus.textContent = pAfter.status;
  el.playerTransition.textContent = pBefore.name !== pAfter.name ? `Started turn as ${pBefore.name}` : "";
  applySprite(el.playerSprite, pAfter.name, true);
  setHp(el.playerHpBar, el.playerHpLabel, pAfter.hp);

  el.opponentName.textContent = oAfter.name;
  el.opponentStatus.textContent = oAfter.status;
  el.opponentTransition.textContent = oBefore.name !== oAfter.name ? `Started turn as ${oBefore.name}` : "";
  applySprite(el.opponentSprite, oAfter.name, false);
  setHp(el.opponentHpBar, el.opponentHpLabel, oAfter.hp);

  el.modelAction.textContent = actionText("Model", turn.player_action);
  el.opponentAction.textContent = actionText("Opponent", turn.opponent_action);
  el.rewardLine.textContent = `Reward: ${Number(turn.reward || 0).toFixed(2)} | Cumulative: ${Number(turn.cumulative_reward || 0).toFixed(2)}`;

  el.commentaryList.innerHTML = "";
  (turn.commentary || []).slice(0, 8).forEach((line) => {
    const li = document.createElement("li");
    li.textContent = line;
    el.commentaryList.appendChild(li);
  });

  renderChips(el.playerTeam, replayState.data.teams.player.map((m) => m.name), pAfter.name);
  const knownOpp = revealedOpponent(turn.turn);
  renderChips(el.opponentTeam, knownOpp, oAfter.name, Math.max(0, 6 - knownOpp.length));
  renderChips(
    el.validActions,
    turn.valid_actions || [],
    null,
    0,
    (action) => `${action.action}: ${action.choice}`
  );
}

function stopAutoplay() {
  if (replayState.autoplayTimer) {
    clearTimeout(replayState.autoplayTimer);
    replayState.autoplayTimer = null;
  }
}

function stepTo(turnNumber) {
  stopAutoplay();
  replayState.frameMode = true;
  renderTurn(turnNumber);
}

function autoplayFrom(turnNumber = 1) {
  stopAutoplay();
  replayState.frameMode = false;
  let index = Math.max(1, turnNumber);

  const tick = () => {
    renderTurn(index);
    if (index >= replayState.data.turns.length) {
      replayState.autoplayTimer = null;
      return;
    }
    const delay = Number(el.speed.value || 2.5) * 1000;
    index += 1;
    replayState.autoplayTimer = setTimeout(tick, delay);
  };

  tick();
}

function activateTab(name) {
  el.tabs.forEach((tab) => tab.classList.toggle("is-active", tab.dataset.tab === name));
  el.panels.forEach((panel) => panel.classList.toggle("is-active", panel.id === `tab-${name}`));
}

async function boot() {
  const res = await fetch("/api/replay");
  replayState.data = await res.json();
  renderSummary();
  activateTab("replay");
}

el.tabs.forEach((tab) => {
  tab.addEventListener("click", () => activateTab(tab.dataset.tab));
});

el.speed.addEventListener("input", () => {
  el.speedValue.textContent = `${Number(el.speed.value).toFixed(1)}s`;
});

el.startBtn.addEventListener("click", () => autoplayFrom(1));
el.frameBtn.addEventListener("click", () => stepTo(1));
el.prevBtn.addEventListener("click", () => stepTo(replayState.currentTurn - 1));
el.nextBtn.addEventListener("click", () => stepTo(replayState.currentTurn + 1));
el.jumpBtn.addEventListener("click", () => stepTo(Number(el.turnInput.value || 1)));
el.turnInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") stepTo(Number(el.turnInput.value || 1));
});

boot();
