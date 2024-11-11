import { FilterQuery, UpdateQuery, QueryOptions } from "mongoose";
import cosumeModel, { CosumeModel } from "../../connect/mongo/model/cosume";

export default class {
    addCosume(info: CosumeModel) {
        return cosumeModel.insert(info);
    }
    updateCosume(info: { filter: FilterQuery<any>; update: UpdateQuery<any>; options?: QueryOptions<any> }) {
        return cosumeModel.updateOne(info);
    }
}
