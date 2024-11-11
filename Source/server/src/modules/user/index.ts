import { Module } from "pigger/core";
import UserControl from "./control";
import UserService from "./service";

@Module({
    controls: [UserControl],
    injects: [UserService],
})
export default class {}
