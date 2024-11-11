import { Model, Schema } from "../../mongo";

interface Params {
    [x: string]: any;
    model: string;
    seed?: number;
    topicNum?: number;
    area?: string;
}
interface Config {
    [x: string]: any;
    filed: Array<string>;
    sort?: string;
    key: number;
}

export interface OrderModel {
    [x: string]: any;
    params: Params;
    config: Config;
    done?: boolean;
    entryTime?: number;
    themeText?: string;
}

const schema = new Schema(
    {
        params: Object,
        config: Object,
        done: Boolean,
        entryTime: Number,
        themeText: String,
    },
    { minimize: false, versionKey: false }
);

export default Model({
    schema,
    collection: "order",
});
