import axios from "axios";
import router from "../router";

const instance = axios.create();
instance.defaults.withCredentials = true;

instance.interceptors.response.use(
    response => {
        if (response.status === 200 && response.data.code === "error" && response.data?.type === "verify") {
            router.push({ path: "/user-info" });
        }
        return response;
    },
    error => Promise.reject(error)
);
export default instance;
