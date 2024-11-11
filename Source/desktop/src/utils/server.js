const { spawn } = require("child_process");
const path = require("path");

let server = null;

async function start() {
    server = spawn("python3", ["app.py"], {
        cwd: path.resolve(__dirname, "../py"),
    });
    server.stdout.on("data", data => {
        console.log(`${data}`);
    });
    server.on("error", err => {
        console.log(`${err}`);
        stop();
        process.exit(1);
    });
}
async function stop() {
    console.log("stop");
    server.kill("SIGINT");
}

module.exports = {
    start,
    stop,
};
