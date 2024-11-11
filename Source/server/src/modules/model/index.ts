import { Module } from "pigger/core";
import ModelControl from "./control";
import ModelService from "./service";
import CosumeService from "../cosume/service";

@Module({
    controls: [ModelControl],
    injects: [ModelService, CosumeService],
})
export default class {}
