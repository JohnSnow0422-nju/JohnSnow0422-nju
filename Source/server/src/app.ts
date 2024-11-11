import Pigger from "pigger";
import { HttpMethodEnum, koaBody } from "koa-body";
import AppModule from "./app.module";
import { redisConnect, mongoConnect } from "./connect";
import { CONFIG, session, verifySession } from "./utils/session";
const koaBodyMiddleware = koaBody({
    multipart: true,
    formidable: {
        maxFileSize: 50 * 1024 * 1024, // 设置上传文件大小最大限制，默认2M
    },
});

const start = async () => {
    const app = new Pigger();
    app.createFactory(AppModule);
    app.keys = ["pigger keys"];
    Promise.all([redisConnect(), mongoConnect()]);
    //inject middleware here
    app.use(koaBodyMiddleware)
        .use(async (ctx, next) => {
            const start = Number(new Date());
            await next();
            const ms = Number(new Date()) - start;
            console.log(`${ctx.method} ${ctx.url} - ${ms}ms`);
        })
        .use(session(CONFIG as any, app))
        .use(verifySession);
    app.routing();
    app.listen(3000);

    app.on("error", (err, ctx) => console.error("server error", err, ctx));
};

start();
