const axios = require("axios");
const url = "http://43.142.133.76";
// const url = "http://localhost:5173";
const fs = require("fs");
const { app } = require("electron");
const path = app.getPath("downloads");
const sleep = time =>
    new Promise(resolve => {
        setTimeout(() => {
            resolve(true);
        }, time);
    });
// const Logger = require("electron-log");

/**
 *
 * @param {Electron.BrowserWindow} win
 * @param {Array<{url:string, response?:Function, request?:Function}>} urlList
 */
function getHttpData(win, urlList) {
    const session = win.webContents.session;
    try {
        win.webContents.debugger.attach("1.1");
    } catch (error) {
        console.log("调试器连接失败: ", err);
    }
    win.webContents.debugger.on("detach", (_event, reason) => {
        console.log("调试器由于以下原因而分离 : ", reason);
    });
    let listenList = [];
    win.webContents.debugger.on("message", async (_event, method, params) => {
        if (method === "Network.requestWillBeSent" && params?.request?.url) {
            const target = urlList
                .filter(item => params.request?.url?.includes(item.url) && params.request.method !== "GET")
                .forEach(api => {
                    if (!listenList.some(item => item?.url === api.url)) listenList.push(api);
                });
        }
        if (method === "Network.responseReceived") {
            const mimeType = params.response.mimeType;
            if (mimeType != "image/gif" && mimeType != "image/jpeg" && mimeType == "application/json") {
                await sleep(1000);
                win.webContents.debugger
                    .sendCommand("Network.getResponseBody", { requestId: params.requestId })
                    .then(async res => {
                        const cookies = await session.cookies.get({});
                        const target = urlList.filter(item => params.response?.url?.includes(item.url));
                        target.forEach(item => {
                            // console.log(res, params);
                            item?.response && item.response(JSON.parse(res.body), cookies);
                        });
                    });
            }
        }
    });
    win.webContents.debugger.sendCommand("Network.enable");
}

async function run(id, cookieList = []) {
    try {
        const cookie = cookieList.map(item => `${item.name}=${item.value}`).join(";");
        const download = () =>
            new Promise(resolve => {
                axios
                    .get(`${url}/model/file?id=${id}&filename=source.xlsx`, {
                        headers: { cookie },
                        responseType: "stream",
                    })
                    .then(res => {
                        if (!fs.existsSync(`${path}/${id}`)) fs.mkdirSync(`${path}/${id}`);
                        const fileStream = fs.createWriteStream(`${path}/${id}/source.xlsx`);
                        res.data.pipe(fileStream);
                        fileStream.on("finish", () => {
                            resolve(true);
                        });
                    })
                    .catch(err => {
                        console.log("download-error", err);
                        // Logger.error("download-error");
                        // Logger.error(err);
                    });
            });
        await download();
        const { data: res } = await axios.get(`${url}/model/info?id=${id}`, { headers: { cookie } });
        const { data, code, msg } = res;
        axios.post("http://localhost:3002/process", { ...data, path }).catch(err => {
            // Logger.info("py-err", err);
            console.log(err);
        });
    } catch (error) {
        console.log(error);
    }
    // console.log("开始跑python", res);
}

/**
 *
 * @param {Electron.BrowserWindow} win
 */
async function runModel(win) {
    try {
        let cookieList = [];
        getHttpData(win, [
            {
                method: "post",
                url: "/model/info",
                response: (res, cookies) => {
                    cookieList = [...cookies];
                    res?.msg >= 80 && res.data && run(res.data, cookies);
                },
            },
        ]);
    } catch (error) {
        console.log(error, "run-model");
    }
}

module.exports = {
    runModel,
    url,
};
