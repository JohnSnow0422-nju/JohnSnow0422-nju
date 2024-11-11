import { Context } from "koa";
import { Control, Get, Post, Put } from "pigger/core";
import path from "path";
import fs from "fs";
import { PassThrough } from "stream";
import axios from "axios";
import ModelService from "./service";
import CosumeService from "../cosume/service";
import { upload, getFileBill, emptyDir, parser, AI_KEY } from "../../utils/common";
import ZhuaYi from "../../utils/pay";

const fileDir = path.resolve(import.meta.dirname, "../../../../output");

@Control("/model")
export default class {
    service: ModelService & CosumeService;

    @Post("/info")
    async uploadInfo(ctx: Context) {
        let orderId: any = -1;
        let dirname = ``;
        let consumeId = -1;
        try {
            const input = ctx.request.files["file"] as any;
            const { params: paramsString, config: configString } = ctx.request.body;
            const params = JSON.parse(paramsString);
            const config = JSON.parse(configString);
            const bill = getFileBill(input.size, params.model);
            const { _doc } = await this.service.addModel({ params, config, done: false, entryTime: Date.now() });
            orderId = _doc._id;
            if (!fs.existsSync(fileDir)) fs.mkdirSync(fileDir);
            dirname = `${fileDir}/${_doc._id}`;
            fs.mkdirSync(dirname);
            const reader = fs.createReadStream(input.filepath); // 创建可读流
            const pay = new ZhuaYi({ user: ctx.session?.username }).consume(ctx.session?.payId, bill, params.model);
            const consume = this.service.addCosume({
                payId: ctx.session?.payId as string,
                userId: ctx.session?.username as string,
                orderId,
                consumeTime: Date.now(),
                bill: bill,
                modelType: params.model,
                fileSize: input.size,
                status: 1,
            });
            const uploadFile = upload(reader, `${dirname}/source.xlsx`);
            const promiseList = await Promise.allSettled([pay, consume, uploadFile]);
            consumeId = (promiseList[1] as any).value?._doc?._id || -1;
            const hasErr = promiseList.find(item => item.status === "rejected");
            if (hasErr) throw new Error(hasErr.reason);
            // axios.post("http://localhost:3002/process", { _id: orderId, ...data }).catch(() => {
            //     console.log(`pyf服务启动失败`);
            // });
            ctx.response.body = {
                msg: bill,
                data: orderId,
                code: "success",
            };
        } catch (error) {
            console.log(error);
            fs.existsSync(dirname) && emptyDir(dirname);
            orderId !== -1 && this.service.removeModel({ _id: orderId });
            consumeId !== -1 && this.service.updateCosume({ filter: { _id: consumeId }, update: { status: -1 } });
            ctx.response.body = {
                msg: `余额不足,请注意充值`,
                code: "error",
            };
        }
    }
    @Put("/info")
    async updateInfo(ctx: Context) {
        try {
            const { filter, update } = ctx.request.body;
            const { data, code, msg } = await this.service.updateModel(filter, update);
            ctx.response.body = {
                data,
                code,
                msg,
            };
        } catch (error) {
            ctx.response.body = {
                code: "error",
                msg: `${error}`,
            };
        }
    }
    @Get("/info")
    async getInfo(ctx: Context) {
        try {
            const { id } = ctx.query;
            const { code, data, msg } = await this.service.getModel({ _id: id });
            ctx.response.body = {
                code,
                data,
                msg,
            };
        } catch (error) {
            ctx.response.body = {
                data: {},
                msg: `${error}`,
                code: "error",
            };
        }
    }
    @Get("/fileList")
    async getFileList(ctx: Context) {
        try {
            const { id } = ctx.request.query;
            const data = fs.readdirSync(`${fileDir}/${id}`);
            ctx.response.body = {
                data,
                code: "success",
                msg: "获取文件列表成功",
            };
        } catch (error) {
            ctx.response.body = {
                code: "error",
                data: [],
                msg: `请输入正确的模型编号`,
            };
        }
    }
    @Get("/file")
    async downloadFile(ctx: Context) {
        const { filename, id } = ctx.query;
        const path = `${fileDir}/${id}/${filename}`;
        const fileStream = fs.createReadStream(path); // 创建可读流
        const stream = fileStream.pipe(new PassThrough());
        // 将文件流作为响应体发送给客户端
        ctx.body = stream;
    }
    @Get("/report")
    async getReport(ctx: Context) {
        try {
            const { id } = ctx.request.query;
            const { code, data: params, msg } = await this.service.getModel({ _id: id });
            if (code === "success" && params?.params?.model && params?._id) {
                const data = parser(params);
                ctx.response.body = {
                    code: "success",
                    data,
                    model: params?.params?.model,
                };
            } else {
                ctx.response.body = {
                    code: "error",
                    msg: "请输入正确的模型编号",
                };
            }
        } catch (error) {
            ctx.response.body = {
                code: "error",
                msg: `报告暂时未生成: ${error}`,
            };
        }
    }
    @Put("/files")
    async updateFiles(ctx: Context) {
        try {
            const fileList = ctx.request.files as any;
            const { id } = ctx.request.body;
            const promiseList: Array<Promise<boolean | Error>> = [];
            Object.keys(fileList).forEach(filename => {
                const file = fileList[filename];
                const reader = fs.createReadStream(file.filepath);
                promiseList.push(upload(reader, `${fileDir}/${id}/${filename}`));
            });
            const res = await Promise.allSettled(promiseList);
            if (res.some(item => !(item as any).value)) throw new Error("未完全上传成功");
            ctx.response.body = {
                code: "success",
                msg: "上传成功",
            };
        } catch (error) {
            ctx.response.body = {
                code: "error",
                msg: `${error}`,
            };
        }
    }
    @Post("/ai")
    async getAnswer(ctx: Context) {
        try {
            const { prompt } = ctx.request.body;
            if (!prompt) {
                ctx.response.body = {
                    code: "error",
                    msg: "请输入有效问题",
                };
                return;
            }

            ctx.set({
                Connection: "keep-alive",
                "Cache-Control": "no-cache",
                "Content-Type": "text/event-stream", // 表示返回数据是个 stream
            });

            const response = await axios.post(
                "https://api.chatanywhere.com.cn/v1/chat/completions",
                //gpt-4o价格有点点贵，先用3.5吧，等用户多了再换
                { model: "gpt-4o", messages: [{ role: "user", content: prompt }], stream: true },
                { headers: { Authorization: `Bearer ${AI_KEY}` }, responseType: "stream" }
            );
            const stream = new PassThrough();
            ctx.body = stream;
            ctx.status = 200;
            response.data.on("data", (chunk: any) => stream.write(chunk));
            response.data.on("end", () => stream.end());
        } catch (error) {
            console.log(error);
            ctx.response.body = {
                code: "error",
                msg: `${error}`,
            };
        }
    }
}
