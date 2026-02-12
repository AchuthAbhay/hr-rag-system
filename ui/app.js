let chats = JSON.parse(localStorage.getItem("chats") || "{}");
let currentChatId = null;

/* ---------- Chat Creation ---------- */

function newChat() {
    const id = "chat_" + Date.now();

    chats[id] = {
        name: "New Chat",
        messages: []
    };

    currentChatId = id;
    saveChats();
    renderChatList();
    renderMessages();
}

function autoNameChat(text) {
    return text.split(" ").slice(0,5).join(" ");
}

/* ---------- Storage ---------- */

function saveChats() {
    localStorage.setItem("chats", JSON.stringify(chats));
}

/* ---------- Sidebar ---------- */

function renderChatList() {

    const div = document.getElementById("chatList");
    div.innerHTML = "";

    Object.entries(chats).forEach(([id, chat]) => {

        const row = document.createElement("div");
        row.className = "chat-row";

        const name = document.createElement("span");
        name.className = "chat-title";
        name.innerText = chat.name;

        name.onclick = () => {
            currentChatId = id;
            renderMessages();
        };

        // ---- menu button (⋯) ----
        const menuBtn = document.createElement("button");
        menuBtn.className = "menu-btn";
        menuBtn.innerText = "⋯";

        menuBtn.onclick = (e) => {
            e.stopPropagation();
            openChatMenu(e, id);
        };

        row.appendChild(name);
        row.appendChild(menuBtn);
        div.appendChild(row);
    });
}


/* ---------- Messages ---------- */

function renderMessages() {
    const win = document.getElementById("chatWindow");
    win.innerHTML = "";

    if (!currentChatId) return;

    chats[currentChatId].messages.forEach(m => {
        addMessage(m.text, m.role, false);
    });
}

function addMessage(text, role, store=true) {
    const win = document.getElementById("chatWindow");

    const d = document.createElement("div");
    d.className = "msg " + role;
    d.innerText = text;

    win.appendChild(d);
    win.scrollTop = win.scrollHeight;

    if (store && currentChatId) {
        chats[currentChatId].messages.push({role,text});
        saveChats();
    }
}

/* ---------- Send Message ---------- */

async function sendMsg() {

    const inp = document.getElementById("msgInput");
    const text = inp.value.trim();
    if (!text) return;

    if (!currentChatId) newChat();

    // auto name first message
    if (chats[currentChatId].messages.length === 0) {
        chats[currentChatId].name = autoNameChat(text);
        renderChatList();
    }

    addMessage(text, "user");
    inp.value = "";   // ✅ clear input

    const thinking = addTempThinking();

    const res = await fetch("http://127.0.0.1:8000/ask", {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body: JSON.stringify({question:text,k:4})
    });

    const data = await res.json();

    thinking.remove();

    addMessage(
        data.answer + "\n\nConfidence: " + data.confidence,
        "bot"
    );
}

function addTempThinking() {
    const win = document.getElementById("chatWindow");
    const d = document.createElement("div");
    d.className = "msg bot";
    d.innerText = "Thinking...";
    win.appendChild(d);
    return d;
}

/* ---------- Upload ---------- */

async function uploadDoc() {

    const file = document.getElementById("fileInput").files[0];
    if (!file) return;

    const fd = new FormData();
    fd.append("file", file);

    const res = await fetch("http://127.0.0.1:8000/upload-doc", {
        method:"POST",
        body: fd
    });

    const data = await res.json();

    document.getElementById("docStatus").innerText =
        "📄 " + data.file + " ✅ indexed";

}


/* ---------- Init ---------- */

renderChatList();

