const { app, BrowserWindow, Menu } = require("electron");
const path = require("path");
const { runModel, url } = require("./src/utils/run");

let win;
function createWindow() {
    try {
        Menu.setApplicationMenu(null);
        win = new BrowserWindow({
            title: "墨斗",
            webPreferences: {
                devTools: false,
                nodeIntegration: true,
                preload: path.resolve(__dirname, "./src/utils/preload.js"),
            },
            icon: "./src/assets/logo.png",
        });
        win.once("ready-to-show", () => win.show());
        win.loadURL(url);
        win.webContents.openDevTools();
    } catch (error) {
        console.log(error);
    }
}

app.whenReady().then(async () => {
    createWindow();
    runModel(win);
});
app.on("window-all-closed", async () => {
    // 在 macOS 上，除非用户用 Cmd + Q 确定地退出，
    // 否则绝大部分应用及其菜单栏会保持激活。()
    if (process.platform !== "darwin") {
        app.quit();
    }
});
app.on("activate", () => {
    // 在macOS上，当单击dock图标并且没有其他窗口打开时，
    // 通常在应用程序中重新创建一个窗口。
    if (BrowserWindow.getAllWindows().length === 0) {
        createWindow();
    }
});
