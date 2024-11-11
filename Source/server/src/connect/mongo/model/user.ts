import { Model, Schema } from "../../mongo";

export interface UserModel {
    username: string;
    password: string;
    registerTime: number;
}

const schema = new Schema(
    {
        username: String,
        password: String,
        registerTime: Number,
    },
    { minimize: false, versionKey: false }
);

export default Model({
    schema,
    collection: "user",
});
