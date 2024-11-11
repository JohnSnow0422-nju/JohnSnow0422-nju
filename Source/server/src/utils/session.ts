import { redis } from "../connect";
import { RedisClientType } from "redis";
import { Context, Next } from "koa";
import { HttpMethodEnum } from "koa-body";

const prefix: string = "auth";

export class SessionStore {
    redis: RedisClientType = redis;
    async get(id: string): Promise<Object> {
        const data = await this.redis.get(`${id}`);
        return JSON.parse(data);
    }
    async set(id: string, data: Object, maxAge: number, { changed, rolling }): Promise<string> {
        const session = JSON.stringify(data);
        await this.redis.set(id, session, { EX: maxAge / 1000 });
        return id;
    }
    async destroy() {}
}

export const CONFIG = {
    prefix: prefix,
    key: "koa:sess",
    httpOnly: true,
    // maxAge: 24 * 60 * 60 * 1000,
    autoCommit: true,
    signed: true,
    renew: true,
    store: new SessionStore(),
};

export { default as session } from "koa-session";

export async function verifySession(ctx: Context, next: Next) {
    try {
        const { url } = ctx.request;
        const requestUrl = url.split("?")[0];
        const skipList = [
            { url: "/user/login", method: HttpMethodEnum.POST },
            { url: "/model/files", method: HttpMethodEnum.PUT },
            { url: "/model/file", method: HttpMethodEnum.GET },
            { url: "/model/info", method: HttpMethodEnum.GET },
        ];
        const target = skipList.find(api => api.url === requestUrl);
        if ((target && target.method === ctx.request.method) || ctx.session?.logged) {
            await next();
            return;
        }
        ctx.response.body = {
            code: "error",
            msg: "用户信息错误, 请重新登陆。",
            type: "verify",
        };
    } catch (error) {
        ctx.response.body = {
            code: "error",
            msg: "请重新登录用户信息",
            type: "verify",
        };
    }
}
