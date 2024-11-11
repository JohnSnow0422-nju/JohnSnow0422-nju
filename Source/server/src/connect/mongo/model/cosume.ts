import { Model, Schema } from "../../mongo";

export interface CosumeModel {
    payId: string;
    userId: string;
    orderId: string;
    consumeTime?: number;
    bill: number;
    modelType?: string;
    fileSize?: number;
    status?: number;
}

const schema = new Schema(
    {
        payId: String,
        userId: String,
        orderId: String,
        consumeTime: Number,
        bill: Number,
        modelType: String,
        fileSize: Number,
        status: Number,
    },
    { minimize: false, versionKey: false }
);

export default Model({
    schema,
    collection: "cosume",
});
