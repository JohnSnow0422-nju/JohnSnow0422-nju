import { createApp } from "vue";
import { createPinia } from "pinia";
import App from "./App.vue";
import "ant-design-vue/dist/reset.css";
import Antdv from "ant-design-vue";
import router from "./router";

const app = createApp(App);

if (window.navigator.userAgent.includes("Electron")) {
    app.use(createPinia()).use(Antdv).use(router).mount("#app");
} else {
    let time = 5;
    setInterval(() => {
        document.getElementById(
            "app"
        ).innerHTML = `${time}秒后跳转至墨斗下载百度云网盘,<a href="https://pan.baidu.com/s/1eEG0qjCdarGsF4bQicts8A?pwd=1234">点击</a>立即跳转, 提取码为1234`;
        time--;
    }, 1000);
    setTimeout(() => {
        window.location = "https://pan.baidu.com/s/1eEG0qjCdarGsF4bQicts8A?pwd=1234";
    }, time * 1000);
}
// https://pan.baidu.com/s/1eEG0qjCdarGsF4bQicts8A?pwd=1234 提取码: 1234
