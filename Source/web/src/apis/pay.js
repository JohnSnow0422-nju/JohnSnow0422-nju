import axios from "./request";
import { userStore } from "../store/user";

async function loginApi(userInfo) {
    try {
        const {
            data: { data, code, msg },
        } = await axios.post("/user/login", userInfo);
        if (code === "success") {
            const store = userStore();
            store.$patch({
                username: data.username,
                payLink: data.payLink,
                payInfo: data.payInfo,
                password: userInfo.password,
            });
        }
        return { data, code, msg };
    } catch (error) {
        console.log(error);
    }
}

export { loginApi };
