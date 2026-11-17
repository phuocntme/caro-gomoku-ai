const BOARD_SIZE = 15;
const WIN_CONDITION = 5;
const TURN_DURATION = 15;
const AI_PLAYER = 'O';
const AI_MODEL_PATH = 'checkpoints/caro_v2_model_15x15.onnx?v=embedded-weights-2';

let currentPlayer = 'X';
let boardState = [];
let gameActive = true;
let lastPlacedCell = null;
let moveHistory = [];
let gameMode = localStorage.getItem('caro_gameMode') === 'ai' ? 'ai' : 'pvp';
let aiSession = null;
let aiLoading = null;
let aiThinking = false;
let aiRequestId = 0;

let scoreX = parseInt(localStorage.getItem('caro_scoreX')) || 0;
let scoreO = parseInt(localStorage.getItem('caro_scoreO')) || 0;
let soundMuted = localStorage.getItem('caro_soundMuted') === 'true';
let timerEnabled = localStorage.getItem('caro_timerEnabled') !== 'false';
let timerId = null;
let timerVal = TURN_DURATION;
let audioCtx = null;

const boardElement = document.getElementById('board');
const turnIndicator = document.getElementById('turnIndicator');
const resetBtn = document.getElementById('resetBtn');
const modalResetBtn = document.getElementById('modalResetBtn');
const winnerModal = document.getElementById('winnerModal');
const winnerText = document.getElementById('winnerText');
const soundToggle = document.getElementById('soundToggle');
const soundOnIcon = document.getElementById('soundOnIcon');
const soundOffIcon = document.getElementById('soundOffIcon');
const timerToggle = document.getElementById('timerToggle');
const resetScoresBtn = document.getElementById('resetScoresBtn');
const undoBtn = document.getElementById('undoBtn');
const scoreXElement = document.getElementById('scoreX');
const scoreOElement = document.getElementById('scoreO');
const playerXLabel = document.getElementById('playerXLabel');
const playerOLabel = document.getElementById('playerOLabel');
const timerContainer = document.getElementById('timerContainer');
const timerBar = document.getElementById('timerBar');
const oneVsOneBtn = document.getElementById('oneVsOneBtn');
const vsAiBtn = document.getElementById('vsAiBtn');
const onlineBtn = document.getElementById('onlineBtn');
const aiStatus = document.getElementById('aiStatus');

let socket = null;
let onlineRoomId = null;
let myOnlineRole = null;

const roomModal = document.getElementById('roomModal');
const closeRoomModalBtn = document.getElementById('closeRoomModalBtn');
const createRoomBtn = document.getElementById('createRoomBtn');
const joinRoomBtn = document.getElementById('joinRoomBtn');
const roomCodeInput = document.getElementById('roomCodeInput');
const roomInitialArea = document.getElementById('roomInitialArea');
const roomStatusArea = document.getElementById('roomStatusArea');
const displayRoomCode = document.getElementById('displayRoomCode');
const copyRoomCodeBtn = document.getElementById('copyRoomCodeBtn');
const roomStatusMsg = document.getElementById('roomStatusMsg');
const roomErrorMsg = document.getElementById('roomErrorMsg');
const canvas = document.getElementById('fireworksCanvas');
const ctx = canvas.getContext('2d');
let particles = [];
let animationFrameId = null;

function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
}

function adjustBoardScale() {
    const margin = 24;
    const maxSize = window.matchMedia('(orientation: landscape)').matches ? 46 : 42;
    const minSize = window.matchMedia('(orientation: landscape)').matches ? 24 : 22;
    const size = Math.max(minSize, Math.min(Math.floor((Math.min(window.innerWidth, window.innerHeight) - margin) / BOARD_SIZE), maxSize));
    document.documentElement.style.setProperty('--cell-size', `${size}px`);
}

function playSound(type) {
    if (soundMuted) return;
    try {
        audioCtx ||= new (window.AudioContext || window.webkitAudioContext)();
        if (audioCtx.state === 'suspended') audioCtx.resume();
        const now = audioCtx.currentTime;
        const notes = { clickX: [640, 'sine', 0.08], clickO: [390, 'triangle', 0.1], warning: [900, 'square', 0.06], reset: [300, 'sine', 0.1], draw: [240, 'sine', 0.25], victory: [660, 'triangle', 0.35] };
        const [frequency, wave, duration] = notes[type] || notes.clickX;
        const oscillator = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        oscillator.type = wave;
        oscillator.frequency.setValueAtTime(frequency, now);
        gain.gain.setValueAtTime(0.08, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + duration);
        oscillator.connect(gain).connect(audioCtx.destination);
        oscillator.start(now);
        oscillator.stop(now + duration);
    } catch (_) {}
}

function startTimer() {
    stopTimer();
    if (!timerEnabled || !gameActive || aiThinking) {
        timerContainer.classList.add('hidden');
        return;
    }
    timerContainer.classList.remove('hidden');
    timerVal = TURN_DURATION;
    timerBar.style.transform = 'scaleX(1)';
    timerId = setInterval(() => {
        timerVal--;
        timerBar.style.transform = `scaleX(${timerVal / TURN_DURATION})`;
        if (timerVal > 0 && timerVal <= 3) playSound('warning');
        if (timerVal <= 0) {
            stopTimer();
            handleTimeOut();
        }
    }, 1000);
}

function stopTimer() {
    if (timerId) clearInterval(timerId);
    timerId = null;
}

function handleTimeOut() {
    if (!gameActive || aiThinking) return;
    currentPlayer = currentPlayer === 'X' ? 'O' : 'X';
    updateGameUI();
    if (isAiTurn()) requestAiMove();
    else startTimer();
}

class Particle {
    constructor(x, y, color) {
        this.x = x; this.y = y; this.color = color;
        this.radius = Math.random() * 3 + 1;
        const angle = Math.random() * Math.PI * 2;
        const speed = Math.random() * 6 + 2;
        this.vx = Math.cos(angle) * speed;
        this.vy = Math.sin(angle) * speed - 2;
        this.alpha = 1;
        this.decay = Math.random() * 0.02 + 0.01;
    }
    update() { this.vy += 0.07; this.x += this.vx; this.y += this.vy; this.alpha -= this.decay; }
    draw() { ctx.save(); ctx.globalAlpha = this.alpha; ctx.fillStyle = this.color; ctx.beginPath(); ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2); ctx.fill(); ctx.restore(); }
}

function startFireworks() {
    const colors = ['#00f0ff', '#ff0055', '#39ff14', '#8b5cf6'];
    for (let i = 0; i < 100; i++) particles.push(new Particle(canvas.width / 2, canvas.height / 2, colors[i % colors.length]));
    animateParticles();
}

function stopFireworks() {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
    particles = [];
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function animateParticles() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    particles = particles.filter((particle) => particle.alpha > 0);
    particles.forEach((particle) => { particle.update(); particle.draw(); });
    if (particles.length) animationFrameId = requestAnimationFrame(animateParticles);
}

function initGame() {
    aiRequestId++;
    aiThinking = false;
    stopTimer();
    stopFireworks();
    boardState = Array.from({ length: BOARD_SIZE }, () => Array(BOARD_SIZE).fill(null));
    currentPlayer = 'X';
    gameActive = true;
    lastPlacedCell = null;
    moveHistory = [];
    hideModal();
    createBoard();
    updateGameUI();
    startTimer();
}

function createBoard() {
    boardElement.innerHTML = '';
    boardElement.style.gridTemplateColumns = `repeat(${BOARD_SIZE}, 1fr)`;
    for (let row = 0; row < BOARD_SIZE; row++) {
        for (let col = 0; col < BOARD_SIZE; col++) {
            const cell = document.createElement('button');
            cell.type = 'button';
            cell.className = 'cell';
            cell.dataset.row = row;
            cell.dataset.col = col;
            cell.setAttribute('aria-label', `Hàng ${row + 1}, cột ${col + 1}`);
            cell.addEventListener('click', handleCellClick);
            boardElement.appendChild(cell);
        }
    }
}

function handleCellClick(event) {
    if (!gameActive || aiThinking || isAiTurn()) return;
    const row = Number(event.currentTarget.dataset.row);
    const col = Number(event.currentTarget.dataset.col);

    if (gameMode === 'online') {
        if (!onlineRoomId) return;
        if (currentPlayer !== myOnlineRole) return;
        if (boardState[row][col] !== null) return;
        const s = initSocket();
        if (s) s.emit('make_move', { row, col });
        return;
    }

    placeMove(row, col);
}

function placeMove(row, col) {
    if (!gameActive || boardState[row][col] !== null) return false;
    const cell = boardElement.children[row * BOARD_SIZE + col];
    if (lastPlacedCell) lastPlacedCell.classList.remove('last-move');
    const player = currentPlayer;
    boardState[row][col] = player;
    moveHistory.push({ row, col, player });
    cell.textContent = player;
    cell.classList.add('occupied', player.toLowerCase(), 'last-move');
    lastPlacedCell = cell;
    playSound(player === 'X' ? 'clickX' : 'clickO');
    if (checkWin(row, col, player)) return true;
    if (isBoardFull()) { handleDraw(); return true; }
    currentPlayer = player === 'X' ? 'O' : 'X';
    updateGameUI();
    if (isAiTurn()) requestAiMove();
    else startTimer();
    return true;
}

function checkWin(row, col, player) {
    const opponent = player === 'X' ? 'O' : 'X';
    for (const [dr, dc] of [[0, 1], [1, 0], [1, 1], [1, -1]]) {
        const cells = [{ row, col }];
        let forwardRow = row + dr; let forwardCol = col + dc;
        while (isPlayerCell(forwardRow, forwardCol, player)) { cells.push({ row: forwardRow, col: forwardCol }); forwardRow += dr; forwardCol += dc; }
        const blockedForward = isInBounds(forwardRow, forwardCol) && boardState[forwardRow][forwardCol] === opponent;
        let backwardRow = row - dr; let backwardCol = col - dc;
        while (isPlayerCell(backwardRow, backwardCol, player)) { cells.push({ row: backwardRow, col: backwardCol }); backwardRow -= dr; backwardCol -= dc; }
        const blockedBackward = isInBounds(backwardRow, backwardCol) && boardState[backwardRow][backwardCol] === opponent;
        if (cells.length >= WIN_CONDITION && !(blockedForward && blockedBackward)) {
            handleWin(cells, player);
            return true;
        }
    }
    return false;
}

function isInBounds(row, col) { return row >= 0 && row < BOARD_SIZE && col >= 0 && col < BOARD_SIZE; }
function isPlayerCell(row, col, player) { return isInBounds(row, col) && boardState[row][col] === player; }
function isBoardFull() { return boardState.every((row) => row.every((cell) => cell !== null)); }

function handleWin(winningCells, player) {
    gameActive = false;
    aiThinking = false;
    stopTimer();
    winningCells.forEach(({ row, col }) => boardElement.children[row * BOARD_SIZE + col].classList.add('winning-cell'));
    if (player === 'X') { scoreX++; localStorage.setItem('caro_scoreX', scoreX); }
    else { scoreO++; localStorage.setItem('caro_scoreO', scoreO); }
    updateScoreboardUI();
    playSound('victory');
    startFireworks();
    let name;
    if (gameMode === 'ai') {
        name = player === 'X' ? 'BẠN' : 'AI';
    } else if (gameMode === 'online') {
        name = player === myOnlineRole ? 'BẠN' : 'ĐỐI THỦ';
    } else {
        name = `PLAYER ${player}`;
    }
    winnerText.innerHTML = `<span class="${player === 'X' ? 'player-x' : 'player-o'}">${name}</span> THẮNG!`;
    setTimeout(showModal, 700);
}

function handleDraw() {
    gameActive = false;
    aiThinking = false;
    stopTimer();
    playSound('draw');
    winnerText.textContent = 'HÒA CỜ!';
    setTimeout(showModal, 600);
}

function updateGameUI() {
    updateTurnIndicator();
    updateActiveScoreCard();
    updateModeUI();
    document.body.dataset.currentPlayer = currentPlayer;
}

function updateTurnIndicator() {
    const playerClass = currentPlayer === 'X' ? 'player-x' : 'player-o';
    let name;
    if (gameMode === 'ai') {
        name = currentPlayer === 'X' ? 'BẠN (X)' : 'AI (O)';
    } else if (gameMode === 'online') {
        name = currentPlayer === myOnlineRole ? `BẠN (${currentPlayer})` : `ĐỐI THỦ (${currentPlayer})`;
    } else {
        name = currentPlayer;
    }
    turnIndicator.innerHTML = aiThinking ? 'AI đang suy nghĩ…' : `Lượt đi: <span class="${playerClass}">${name}</span>`;
}

function updateActiveScoreCard() {
    document.querySelectorAll('.score-card').forEach((card) => card.classList.remove('active-card'));
    if (gameActive) document.querySelector(currentPlayer === 'X' ? '.player-x-card' : '.player-o-card').classList.add('active-card');
}

function updateModeUI() {
    oneVsOneBtn.classList.toggle('active', gameMode === 'pvp');
    vsAiBtn.classList.toggle('active', gameMode === 'ai');
    onlineBtn.classList.toggle('active', gameMode === 'online');
    if (gameMode === 'ai') {
        playerXLabel.textContent = 'BẠN (X)';
        playerOLabel.textContent = 'AI (O)';
    } else if (gameMode === 'online') {
        playerXLabel.textContent = myOnlineRole === 'X' ? 'BẠN (X)' : 'ĐỐI THỦ (X)';
        playerOLabel.textContent = myOnlineRole === 'O' ? 'BẠN (O)' : 'ĐỐI THỦ (O)';
    } else {
        playerXLabel.textContent = 'PLAYER X';
        playerOLabel.textContent = 'PLAYER O';
    }
    if (gameMode !== 'ai' && gameMode !== 'online') aiStatus.textContent = '';
    undoBtn.style.display = gameMode === 'online' ? 'none' : 'inline-flex';
}

function updateScoreboardUI() { scoreXElement.textContent = scoreX; scoreOElement.textContent = scoreO; }

async function loadAiModel() {
    if (aiSession) return aiSession;
    if (!window.ort) throw new Error('Không tải được ONNX Runtime Web.');
    aiLoading ||= window.ort.InferenceSession.create(AI_MODEL_PATH, { executionProviders: ['wasm'] })
        .then((session) => { aiSession = session; return session; })
        .finally(() => { aiLoading = null; });
    return aiLoading;
}

function createAiInput() {
    const planeSize = BOARD_SIZE * BOARD_SIZE;
    const data = new Float32Array(4 * planeSize);
    for (let row = 0; row < BOARD_SIZE; row++) {
        for (let col = 0; col < BOARD_SIZE; col++) {
            const index = row * BOARD_SIZE + col;
            if (boardState[row][col] === currentPlayer) data[index] = 1;
            if (boardState[row][col] === (currentPlayer === 'X' ? 'O' : 'X')) data[planeSize + index] = 1;
        }
    }
    if (moveHistory.length) {
        const { row, col } = moveHistory.at(-1);
        data[2 * planeSize + row * BOARD_SIZE + col] = 1;
    }
    if (currentPlayer === 'X') data.fill(1, 3 * planeSize, 4 * planeSize);
    return new window.ort.Tensor('float32', data, [1, 4, BOARD_SIZE, BOARD_SIZE]);
}

function isWinningMove(row, col, player, board = boardState) {
    const opponent = player === 'X' ? 'O' : 'X';
    for (const [dr, dc] of [[0, 1], [1, 0], [1, 1], [1, -1]]) {
        let count = 1;
        let forwardRow = row + dr;
        let forwardCol = col + dc;
        while (isInBounds(forwardRow, forwardCol) && board[forwardRow][forwardCol] === player) {
            count++;
            forwardRow += dr;
            forwardCol += dc;
        }
        const blockedForward = isInBounds(forwardRow, forwardCol) && board[forwardRow][forwardCol] === opponent;
        let backwardRow = row - dr;
        let backwardCol = col - dc;
        while (isInBounds(backwardRow, backwardCol) && board[backwardRow][backwardCol] === player) {
            count++;
            backwardRow -= dr;
            backwardCol -= dc;
        }
        const blockedBackward = isInBounds(backwardRow, backwardCol) && board[backwardRow][backwardCol] === opponent;
        if (count >= WIN_CONDITION && !(blockedForward && blockedBackward)) return true;
    }
    return false;
}

function createsOpenFour(row, col, player, board = boardState) {
    for (const [dr, dc] of [[0, 1], [1, 0], [1, 1], [1, -1]]) {
        let count = 1;
        let forwardRow = row + dr;
        let forwardCol = col + dc;
        while (isInBounds(forwardRow, forwardCol) && board[forwardRow][forwardCol] === player) {
            count++;
            forwardRow += dr;
            forwardCol += dc;
        }
        let backwardRow = row - dr;
        let backwardCol = col - dc;
        while (isInBounds(backwardRow, backwardCol) && board[backwardRow][backwardCol] === player) {
            count++;
            backwardRow -= dr;
            backwardCol -= dc;
        }
        if (count === WIN_CONDITION - 1 && isInBounds(forwardRow, forwardCol) && isInBounds(backwardRow, backwardCol)
            && board[forwardRow][forwardCol] === null && board[backwardRow][backwardCol] === null) return true;
    }
    return false;
}

function createsAnyFour(row, col, player, board = boardState) {
    const opponent = player === 'X' ? 'O' : 'X';
    for (const [dr, dc] of [[0, 1], [1, 0], [1, 1], [1, -1]]) {
        let count = 1;
        let forwardRow = row + dr;
        let forwardCol = col + dc;
        while (isInBounds(forwardRow, forwardCol) && board[forwardRow][forwardCol] === player) {
            count++;
            forwardRow += dr;
            forwardCol += dc;
        }
        const blockedForward = isInBounds(forwardRow, forwardCol) && board[forwardRow][forwardCol] === opponent;
        let backwardRow = row - dr;
        let backwardCol = col - dc;
        while (isInBounds(backwardRow, backwardCol) && board[backwardRow][backwardCol] === player) {
            count++;
            backwardRow -= dr;
            backwardCol -= dc;
        }
        const blockedBackward = isInBounds(backwardRow, backwardCol) && board[backwardRow][backwardCol] === opponent;
        if (count === WIN_CONDITION - 1 && !(blockedForward && blockedBackward)) return true;
    }
    return false;
}

function createsOpenThree(row, col, player, board = boardState) {
    for (const [dr, dc] of [[0, 1], [1, 0], [1, 1], [1, -1]]) {
        let count = 1;
        let forwardRow = row + dr;
        let forwardCol = col + dc;
        while (isInBounds(forwardRow, forwardCol) && board[forwardRow][forwardCol] === player) {
            count++;
            forwardRow += dr;
            forwardCol += dc;
        }
        let backwardRow = row - dr;
        let backwardCol = col - dc;
        while (isInBounds(backwardRow, backwardCol) && board[backwardRow][backwardCol] === player) {
            count++;
            backwardRow -= dr;
            backwardCol -= dc;
        }
        if (count === WIN_CONDITION - 2 && isInBounds(forwardRow, forwardCol) && isInBounds(backwardRow, backwardCol)
            && board[forwardRow][forwardCol] === null && board[backwardRow][backwardCol] === null) return true;
    }
    return false;
}

const MCTS_SIMULATIONS = 80;
const MCTS_C_PUCT = 1.5;

class MctsNode {
    constructor(prior = 0) {
        this.prior = prior;
        this.children = new Map();
        this.visitCount = 0;
        this.valueSum = 0;
    }

    get q() {
        return this.visitCount ? this.valueSum / this.visitCount : 0;
    }
}

function createMctsInput(board, player, lastMove) {
    const planeSize = BOARD_SIZE * BOARD_SIZE;
    const data = new Float32Array(4 * planeSize);
    const opponent = player === 'X' ? 'O' : 'X';
    for (let row = 0; row < BOARD_SIZE; row++) {
        for (let col = 0; col < BOARD_SIZE; col++) {
            const index = row * BOARD_SIZE + col;
            if (board[row][col] === player) data[index] = 1;
            if (board[row][col] === opponent) data[planeSize + index] = 1;
        }
    }
    if (lastMove) data[2 * planeSize + lastMove.row * BOARD_SIZE + lastMove.col] = 1;
    if (player === 'X') data.fill(1, 3 * planeSize, 4 * planeSize);
    return new window.ort.Tensor('float32', data, [1, 4, BOARD_SIZE, BOARD_SIZE]);
}

function legalActions(board) {
    const actions = [];
    for (let index = 0; index < BOARD_SIZE * BOARD_SIZE; index++) {
        if (board[Math.floor(index / BOARD_SIZE)][index % BOARD_SIZE] === null) actions.push(index);
    }
    return actions;
}

function tacticalBonus(board, row, col, player) {
    const opponent = player === 'X' ? 'O' : 'X';
    board[row][col] = player;
    let ownScore = 0;
    if (isWinningMove(row, col, player, board)) ownScore = 15;
    else if (createsOpenFour(row, col, player, board)) ownScore = 10;
    else if (createsAnyFour(row, col, player, board)) ownScore = 6;
    else if (createsOpenThree(row, col, player, board)) ownScore = 4;

    board[row][col] = opponent;
    let opponentScore = 0;
    if (isWinningMove(row, col, opponent, board)) opponentScore = 12;
    else if (createsOpenFour(row, col, opponent, board)) opponentScore = 8;
    else if (createsAnyFour(row, col, opponent, board)) opponentScore = 5;
    else if (createsOpenThree(row, col, opponent, board)) opponentScore = 3;

    board[row][col] = null;
    return Math.max(ownScore, opponentScore);
}

function findImmediateTacticalMove(board, player) {
    const legal = legalActions(board);
    let bestAction = null;
    let bestScore = -1;
    for (const action of legal) {
        const row = Math.floor(action / BOARD_SIZE);
        const col = action % BOARD_SIZE;
        const score = tacticalBonus(board, row, col, player);
        if (score >= 8 && score > bestScore) {
            bestScore = score;
            bestAction = action;
        }
    }
    return bestScore >= 8 ? bestAction : null;
}

async function evaluateMcts(session, board, player, lastMove) {
    const legal = legalActions(board);
    const output = await session.run({ board_state: createMctsInput(board, player, lastMove) });
    const logits = output.policy_logits.data;
    const bonuses = new Map();
    let maxLogit = -Infinity;
    for (const action of legal) {
        const row = Math.floor(action / BOARD_SIZE);
        const col = action % BOARD_SIZE;
        const value = logits[action] + tacticalBonus(board, row, col, player);
        bonuses.set(action, value);
        maxLogit = Math.max(maxLogit, value);
    }
    let total = 0;
    const priors = new Map();
    for (const action of legal) {
        const prior = Math.exp(bonuses.get(action) - maxLogit);
        priors.set(action, prior);
        total += prior;
    }
    for (const action of legal) priors.set(action, priors.get(action) / total);
    return { priors, value: output.value.data[0] };
}

async function chooseMctsMove(session, requestId) {
    const rootBoard = boardState.map((row) => [...row]);

    // Nước cờ chiến thuật khẩn cấp: Thắng ngay (15), Chặn thắng (12), Tạo 4 mở (10), Chặn 4 mở (8)
    const forced = findImmediateTacticalMove(rootBoard, AI_PLAYER);
    if (forced !== null) return forced;

    const root = new MctsNode();
    const rootEvaluation = await evaluateMcts(session, rootBoard, AI_PLAYER, moveHistory.at(-1));
    for (const [action, prior] of rootEvaluation.priors) root.children.set(action, new MctsNode(prior));

    for (let simulation = 0; simulation < MCTS_SIMULATIONS; simulation++) {
        if (requestId !== aiRequestId) return null;
        const board = rootBoard.map((row) => [...row]);
        let player = AI_PLAYER;
        let lastMove = moveHistory.at(-1) || null;
        let node = root;
        const path = [node];
        let value;

        while (node.children.size) {
            let selectedAction = null;
            let selectedNode = null;
            let bestScore = -Infinity;
            const scale = Math.sqrt(node.visitCount + 1);
            for (const [action, child] of node.children) {
                const score = -child.q + MCTS_C_PUCT * child.prior * scale / (1 + child.visitCount);
                if (score > bestScore) {
                    bestScore = score;
                    selectedAction = action;
                    selectedNode = child;
                }
            }
            node = selectedNode;
            const row = Math.floor(selectedAction / BOARD_SIZE);
            const col = selectedAction % BOARD_SIZE;
            board[row][col] = player;
            lastMove = { row, col };
            path.push(node);
            if (isWinningMove(row, col, player, board)) {
                value = 1;
                break;
            }
            player = player === 'X' ? 'O' : 'X';
        }

        if (value === undefined) {
            const legal = legalActions(board);
            if (!legal.length) value = 0;
            else {
                const evaluation = await evaluateMcts(session, board, player, lastMove);
                value = evaluation.value;
                for (const [action, prior] of evaluation.priors) node.children.set(action, new MctsNode(prior));
            }
        }
        for (let index = path.length - 1; index >= 0; index--) {
            path[index].visitCount++;
            path[index].valueSum += value;
            value = -value;
        }
        if ((simulation + 1) % 10 === 0) {
            aiStatus.textContent = `AI đang suy nghĩ… ${simulation + 1}/${MCTS_SIMULATIONS}`;
            await new Promise((resolve) => setTimeout(resolve, 0));
        }
    }

    let bestAction = null;
    let bestVisits = -1;
    for (const [action, child] of root.children) {
        if (child.visitCount > bestVisits) {
            bestVisits = child.visitCount;
            bestAction = action;
        }
    }
    return bestAction;
}

async function requestAiMove() {
    if (!gameActive || !isAiTurn()) return;
    aiThinking = true;
    stopTimer();
    updateGameUI();
    aiStatus.textContent = 'Đang tải mô hình AI…';
    const requestId = ++aiRequestId;
    try {
        const session = await loadAiModel();
        if (requestId !== aiRequestId || !gameActive || !isAiTurn()) return;
        aiStatus.textContent = `AI đang suy nghĩ… 0/${MCTS_SIMULATIONS}`;
        const move = await chooseMctsMove(session, requestId);
        if (requestId !== aiRequestId || move === null || !gameActive || !isAiTurn()) return;
        aiThinking = false;
        aiStatus.textContent = 'AI sẵn sàng';
        placeMove(Math.floor(move / BOARD_SIZE), move % BOARD_SIZE);
    } catch (error) {
        aiThinking = false;
        aiStatus.textContent = 'Không thể tải AI. Chuyển sang 1 VS 1.';
        gameMode = 'pvp';
        localStorage.setItem('caro_gameMode', gameMode);
        updateGameUI();
        startTimer();
        console.error(error);
    }
}

function isAiTurn() { return gameMode === 'ai' && currentPlayer === AI_PLAYER; }

function showRoomModal() {
    roomModal.style.display = 'flex';
    roomModal.offsetHeight;
    roomModal.classList.add('show');
    roomInitialArea.style.display = 'flex';
    roomStatusArea.style.display = 'none';
    roomErrorMsg.style.display = 'none';
    roomCodeInput.value = '';
}

function hideRoomModal() {
    roomModal.classList.remove('show');
    setTimeout(() => {
        if (!roomModal.classList.contains('show')) roomModal.style.display = 'none';
    }, 400);
}

function setCreateBtnLoading(loading) {
    createRoomBtn.disabled = loading;
    const span = createRoomBtn.querySelector('span');
    if (span) span.textContent = loading ? 'ĐANG KẾT NỐI…' : 'TẠO PHÒNG MỚI';
}

function setJoinBtnLoading(loading) {
    joinRoomBtn.disabled = loading;
    joinRoomBtn.textContent = loading ? 'ĐANG VÀO…' : 'VÀO PHÒNG';
}

function initSocket() {
    if (socket) return socket;
    if (typeof io === 'undefined') {
        roomErrorMsg.textContent = 'Không tìm thấy thư viện kết nối (Socket.io). Vui lòng tải lại trang!';
        roomErrorMsg.style.display = 'block';
        return null;
    }

    socket = io({
        transports: ['polling', 'websocket'],
        reconnectionAttempts: 5,
        timeout: 10000
    });

    socket.on('connect', () => {
        if (roomErrorMsg && roomErrorMsg.textContent.includes('kết nối')) {
            roomErrorMsg.style.display = 'none';
        }
    });

    socket.on('connect_error', (error) => {
        console.error('Socket connect error:', error);
        roomErrorMsg.textContent = 'Không thể kết nối máy chủ online: ' + (error.message || 'Mất kết nối');
        roomErrorMsg.style.display = 'block';
        setCreateBtnLoading(false);
        setJoinBtnLoading(false);
    });

    socket.on('room_created', ({ roomId, role }) => {
        setCreateBtnLoading(false);
        onlineRoomId = roomId;
        myOnlineRole = role;
        roomInitialArea.style.display = 'none';
        roomStatusArea.style.display = 'flex';
        displayRoomCode.textContent = roomId;
        roomStatusMsg.textContent = 'Đang chờ đối thủ tham gia...';
        roomErrorMsg.style.display = 'none';
        aiStatus.textContent = `Phòng: ${roomId} | Bạn là quân ${role} (Đi trước)`;
        updateGameUI();
    });

    socket.on('room_joined', ({ roomId, role }) => {
        setJoinBtnLoading(false);
        onlineRoomId = roomId;
        myOnlineRole = role;
        hideRoomModal();
        aiStatus.textContent = `Phòng: ${roomId} | Bạn là quân ${role} (Đi sau)`;
        updateGameUI();
    });

    socket.on('join_error', (msg) => {
        setJoinBtnLoading(false);
        roomErrorMsg.textContent = msg;
        roomErrorMsg.style.display = 'block';
    });

    socket.on('game_start', ({ roomId }) => {
        hideRoomModal();
        initGame();
        aiStatus.textContent = `Phòng: ${roomId} | Bạn: Quân ${myOnlineRole}`;
        updateGameUI();
    });

    socket.on('move_made', ({ row, col, player, nextTurn }) => {
        if (gameMode !== 'online') return;
        placeMove(row, col);
    });

    socket.on('game_restarted', () => {
        if (gameMode !== 'online') return;
        initGame();
    });

    socket.on('player_left', ({ role, message }) => {
        if (gameMode !== 'online') return;
        alert(message || 'Đối thủ đã thoát phòng!');
        onlineRoomId = null;
        myOnlineRole = null;
        setGameMode('pvp');
    });

    return socket;
}

function setGameMode(mode) {
    if (mode === 'online') {
        gameMode = 'online';
        localStorage.setItem('caro_gameMode', 'online');
        updateModeUI();
        initSocket();
        showRoomModal();
        return;
    }

    if (gameMode === 'online' && mode !== 'online') {
        onlineRoomId = null;
        myOnlineRole = null;
    }

    gameMode = mode;
    localStorage.setItem('caro_gameMode', gameMode);
    initGame();
    if (gameMode === 'ai') {
        aiStatus.textContent = 'Đang chuẩn bị AI…';
        loadAiModel().then(() => { if (gameMode === 'ai') aiStatus.textContent = 'AI sẵn sàng'; }).catch(() => {});
    }
}

function undoMove() {
    if (gameMode === 'online' || aiThinking) return;
    const movesToUndo = gameMode === 'ai' && moveHistory.length >= 2 ? 2 : 1;
    if (!moveHistory.length) return;
    aiRequestId++;
    const wasFinished = !gameActive;
    if (wasFinished) {
        gameActive = true;
        hideModal();
        stopFireworks();
        const winner = moveHistory.at(-1)?.player;
        if (winner === 'X') { scoreX = Math.max(0, scoreX - 1); localStorage.setItem('caro_scoreX', scoreX); }
        if (winner === 'O') { scoreO = Math.max(0, scoreO - 1); localStorage.setItem('caro_scoreO', scoreO); }
    }
    for (let i = 0; i < movesToUndo && moveHistory.length; i++) {
        const move = moveHistory.pop();
        boardState[move.row][move.col] = null;
        const cell = boardElement.children[move.row * BOARD_SIZE + move.col];
        cell.textContent = '';
        cell.className = 'cell';
    }
    const lastMove = moveHistory.at(-1);
    lastPlacedCell = lastMove ? boardElement.children[lastMove.row * BOARD_SIZE + lastMove.col] : null;
    if (lastPlacedCell) lastPlacedCell.classList.add('last-move');
    currentPlayer = 'X';
    if (gameMode !== 'ai' && moveHistory.length) currentPlayer = moveHistory.at(-1).player === 'X' ? 'O' : 'X';
    updateScoreboardUI();
    updateGameUI();
    startTimer();
}

function showModal() { winnerModal.style.display = 'flex'; winnerModal.offsetHeight; winnerModal.classList.add('show'); }
function hideModal() { winnerModal.classList.remove('show'); setTimeout(() => { if (!winnerModal.classList.contains('show')) winnerModal.style.display = 'none'; }, 400); }
function updateSoundButtonUI() { soundToggle.classList.toggle('muted', soundMuted); soundOnIcon.style.display = soundMuted ? 'none' : 'block'; soundOffIcon.style.display = soundMuted ? 'block' : 'none'; }
function updateTimerButtonUI() { timerToggle.classList.toggle('active', timerEnabled); }

function handleGameRestart() {
    playSound('reset');
    if (gameMode === 'online' && onlineRoomId && socket) {
        socket.emit('request_restart');
    } else {
        initGame();
    }
}

soundToggle.addEventListener('click', () => { soundMuted = !soundMuted; localStorage.setItem('caro_soundMuted', soundMuted); updateSoundButtonUI(); if (!soundMuted) playSound('clickX'); });
timerToggle.addEventListener('click', () => { timerEnabled = !timerEnabled; localStorage.setItem('caro_timerEnabled', timerEnabled); updateTimerButtonUI(); if (timerEnabled) startTimer(); else { stopTimer(); timerContainer.classList.add('hidden'); } });
resetScoresBtn.addEventListener('click', () => { scoreX = 0; scoreO = 0; localStorage.setItem('caro_scoreX', 0); localStorage.setItem('caro_scoreO', 0); updateScoreboardUI(); playSound('reset'); });
oneVsOneBtn.addEventListener('click', () => setGameMode('pvp'));
vsAiBtn.addEventListener('click', () => setGameMode('ai'));
onlineBtn.addEventListener('click', () => setGameMode('online'));
undoBtn.addEventListener('click', undoMove);
resetBtn.addEventListener('click', handleGameRestart);
modalResetBtn.addEventListener('click', handleGameRestart);

closeRoomModalBtn.addEventListener('click', () => {
    hideRoomModal();
    if (!onlineRoomId) setGameMode('pvp');
});

createRoomBtn.addEventListener('click', () => {
    roomErrorMsg.style.display = 'none';
    setCreateBtnLoading(true);
    const s = initSocket();
    if (!s) {
        setCreateBtnLoading(false);
        return;
    }
    if (s.connected) {
        s.emit('create_room');
    } else {
        s.once('connect', () => s.emit('create_room'));
    }
});

joinRoomBtn.addEventListener('click', () => {
    const code = roomCodeInput.value.trim().toUpperCase();
    if (!code || code.length < 4) {
        roomErrorMsg.textContent = 'Vui lòng nhập mã phòng hợp lệ!';
        roomErrorMsg.style.display = 'block';
        return;
    }
    roomErrorMsg.style.display = 'none';
    setJoinBtnLoading(true);
    const s = initSocket();
    if (!s) {
        setJoinBtnLoading(false);
        return;
    }
    if (s.connected) {
        s.emit('join_room', code);
    } else {
        s.once('connect', () => s.emit('join_room', code));
    }
});

roomCodeInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') joinRoomBtn.click();
});

copyRoomCodeBtn.addEventListener('click', () => {
    const code = displayRoomCode.textContent;
    if (!code || code === '------') return;

    function showSuccess() {
        copyRoomCodeBtn.textContent = 'ĐÃ CHÉP!';
        setTimeout(() => { copyRoomCodeBtn.textContent = 'SAO CHÉP'; }, 2000);
    }

    if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(code).then(showSuccess).catch(() => fallbackCopy(code));
    } else {
        fallbackCopy(code);
    }

    function fallbackCopy(text) {
        try {
            const input = document.createElement('input');
            input.value = text;
            input.style.position = 'fixed';
            input.style.opacity = '0';
            document.body.appendChild(input);
            input.focus();
            input.select();
            const successful = document.execCommand('copy');
            document.body.removeChild(input);
            if (successful) showSuccess();
            else prompt('Mã phòng của bạn là:', text);
        } catch (_) {
            prompt('Mã phòng của bạn là:', text);
        }
    }
});

window.addEventListener('resize', () => { resizeCanvas(); adjustBoardScale(); });

resizeCanvas();
adjustBoardScale();
updateSoundButtonUI();
updateTimerButtonUI();
initGame();
if (gameMode === 'ai') { aiStatus.textContent = 'Đang chuẩn bị AI…'; loadAiModel().then(() => { if (gameMode === 'ai') aiStatus.textContent = 'AI sẵn sàng'; }).catch(() => {}); }