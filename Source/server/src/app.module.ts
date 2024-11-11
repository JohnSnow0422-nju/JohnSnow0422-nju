import { Module } from "pigger/core";
import UserModule from "./modules/user";
import ModelModule from "./modules/model";

@Module({
    modules: [UserModule, ModelModule],
})
export default class {}
