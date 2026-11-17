const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const path = require('path');
const os = require('os');

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
    cors: {
        origin: '*',
        methods: ['GET', 'POST']
    },
    transports: ['polling', 'websocket'],
    allowEIO3: true
});

const PORT = process.env.PORT || 8080;

// Phục vụ cache dài hạn cho checkpoint ONNX
app.use('/checkpoints', express.static(path.join(__dirname, 'checkpoints'), {
    maxAge: '1y',
    immutable: true
}));

// Phục vụ tài nguyên tĩnh của ứng dụng Caro
app.use(express.static(path.join(__dirname)));

app.get('/health', (req, res) => {
    res.json({ status: 'ok', active_rooms: rooms.size });
});

// Quản lý phòng chơi trong bộ nhớ
const rooms = new Map();

function generateRoomCode() {
    const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
    let code = '';
    for (let i = 0; i < 6; i++) {
        code += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return code;
}

io.on('connection', (socket) => {
    // 1. Tạo phòng mới
    socket.on('create_room', () => {
        let code;
        do {
            code = generateRoomCode();
        } while (rooms.has(code));

        rooms.set(code, {
            id: code,
            players: [{ id: socket.id, role: 'X' }],
            board: Array.from({ length: 15 }, () => Array(15).fill(null)),
            currentTurn: 'X',
            gameActive: false,
            moveHistory: []
        });

        socket.join(code);
        socket.roomId = code;
        socket.emit('room_created', { roomId: code, role: 'X' });
    });

    // 2. Tham gia phòng
    socket.on('join_room', (rawCode) => {
        const code = String(rawCode || '').trim().toUpperCase();
        const room = rooms.get(code);

        if (!room) {
            socket.emit('join_error', 'Mã phòng không tồn tại hoặc đã bị hủy!');
            return;
        }

        if (room.players.length >= 2) {
            socket.emit('join_error', 'Phòng này đã đủ 2 người chơi!');
            return;
        }

        room.players.push({ id: socket.id, role: 'O' });
        room.gameActive = true;
        socket.join(code);
        socket.roomId = code;

        socket.emit('room_joined', { roomId: code, role: 'O' });

        // Thông báo cho cả 2 người trận đấu bắt đầu
        io.to(code).emit('game_start', {
            roomId: code,
            currentTurn: room.currentTurn,
            players: room.players.map((p) => ({ role: p.role }))
        });
    });

    // 3. Thực hiện nước đi
    socket.on('make_move', ({ row, col }) => {
        if (!socket.roomId) return;
        const room = rooms.get(socket.roomId);
        if (!room || !room.gameActive) return;

        const player = room.players.find((p) => p.id === socket.id);
        if (!player || player.role !== room.currentTurn) return;

        if (row < 0 || row >= 15 || col < 0 || col >= 15 || room.board[row][col] !== null) return;

        room.board[row][col] = player.role;
        room.moveHistory.push({ row, col, player: player.role });
        room.currentTurn = player.role === 'X' ? 'O' : 'X';

        io.to(socket.roomId).emit('move_made', {
            row,
            col,
            player: player.role,
            nextTurn: room.currentTurn
        });
    });

    // 4. Khởi động lại ván cờ mới trong phòng
    socket.on('request_restart', () => {
        if (!socket.roomId) return;
        const room = rooms.get(socket.roomId);
        if (!room) return;

        room.board = Array.from({ length: 15 }, () => Array(15).fill(null));
        room.currentTurn = 'X';
        room.gameActive = true;
        room.moveHistory = [];

        io.to(socket.roomId).emit('game_restarted', { currentTurn: 'X' });
    });

    // 5. Thoát phòng hoặc mất kết nối
    socket.on('disconnect', () => {
        if (!socket.roomId) return;
        const room = rooms.get(socket.roomId);
        if (!room) return;

        const leavingPlayer = room.players.find((p) => p.id === socket.id);
        const leavingRole = leavingPlayer ? leavingPlayer.role : 'Người chơi';

        socket.to(socket.roomId).emit('player_left', {
            role: leavingRole,
            message: `Đối thủ (${leavingRole}) đã rời phòng hoặc mất kết nối!`
        });

        rooms.delete(socket.roomId);
    });
});

server.listen(PORT, '0.0.0.0', () => {
    console.log(`====================================================`);
    console.log(`  Caro Neon Online Server is running!`);
    console.log(`  - Local:   http://localhost:${PORT}`);

    const ifaces = os.networkInterfaces();
    for (const dev in ifaces) {
        ifaces[dev].forEach((details) => {
            if (details.family === 'IPv4' && !details.internal) {
                console.log(`  - Mạng ${dev.padEnd(8)}: http://${details.address}:${PORT}`);
            }
        });
    }
    console.log(`====================================================`);
});
