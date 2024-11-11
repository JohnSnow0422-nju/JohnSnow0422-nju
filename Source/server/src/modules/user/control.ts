import { Control, Get, Post } from "pigger/core";
import UserService from "./service";
import { Context } from "koa";
import ZhuaYi from "../../utils/pay";

@Control("/user")
export default class {
    service: UserService;
    @Post("/login")
    async user(ctx: Context) {
        try {
            const { username, password } = ctx.request.body;
            const { data: target } = await this.service.getByUsername({ username });
            if (target && target?.password !== password) {
                ctx.response.body = {
                    code: "error",
                    msg: "密码错误",
                };
                return;
            }
            let userId = username;
            if (!target) {
                const { _doc } = await this.service.add({ username, password, registerTime: Date.now() });
                userId = _doc.username;
            }
            const pay = await new ZhuaYi({ user: userId }).getUserData();
            const payInfo = {
                payLink: pay.home_link?.url,
                payId: pay.id,
            };
            ctx.session = { username, password, logged: true, ...payInfo } as any;
            ctx.response.body = {
                code: "success",
                msg: "获取信息成功",
                data: {
                    username,
                    ...payInfo,
                },
            };
        } catch (error) {
            console.log(error);
        }
    }
}
