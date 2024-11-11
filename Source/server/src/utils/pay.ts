import jwt from "jsonwebtoken";
import axios from "axios";

export default class ZhuaYi {
    BASE_URL = "https://revenue.ezboti.com/api/v1/server";
    projectId = "b37v1bbhsb7hs"; // 艺爪付费项目ID
    projectSecret = "bh95ckkbkext31bdfda93zrry19ax6fq"; // 艺爪付费项目密钥
    user = "";
    payWall = "an7sh3f73mvh6";
    constructor({ user }) {
        this.user = user;
    }
    decodeToken(token) {
        const decoded = jwt.verify(token, this.projectSecret, { complete: true });
        const { result } = decoded.payload;
        return result;
    }
    encodeToken(payload) {
        payload.exp = Date.now() + 30 * 60; // 过期时间，建议当前时间+30分钟
        payload.nonce = Math.random().toString(36).slice(-6); // 随机字符串，32个字符以内
        const header = { project_id: this.projectId };
        return jwt.sign(payload, this.projectSecret, { header });
    }

    async getUserData() {
        const api = "customer.info";
        const params = {
            paywall_id: this.payWall, // 付费界面ID，ID和别名传一个即可
            customer: {
                external_id: this.user, // 商户系统用户ID
            },
            include_balance: true, // 是否返回用户余额
        };
        const token = this.encodeToken({ method: api, params: params });
        const url = `${this.BASE_URL}/${api}`;
        const res = await this._sendRequest({ url, content: token });
        return this.decodeToken(res.data);
    }

    async consume(customerId, amount, model = "") {
        try {
            const api = "customer.consume";
            const userInfo = await this.getUserData();
            const balanceLogId = userInfo?.["balance_s"]?.[0]?.["balance_log_id"];
            const params = {
                customer_id: customerId, // 艺爪系统用户ID
                equity_id: "kuhr15eqnsggw", // 权益ID
                amount, // 数量
                title: `购买${model}模型`, // 标题
                change_type: 16, // 变更类型枚举值
                balance_log_id: balanceLogId, // 最后一次余额更新日志ID
            };
            const token = this.encodeToken({ method: api, params: params });
            const url = `${this.BASE_URL}/${api}`;
            const res = await this._sendRequest({ url, content: token });
            return this.decodeToken(res.data);
        } catch (error) {
            throw new Error(error.response?.data?.error || "消费积分时出现未知错误");
        }
    }

    _sendRequest({ url, content }) {
        return axios({
            method: "POST",
            url: url,
            headers: { "Content-Type": "text/plain" },
            data: content,
        });
    }
}
