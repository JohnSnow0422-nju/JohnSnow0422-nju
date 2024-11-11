import { FilterQuery, UpdateQuery } from "mongoose";
import modelModel, { OrderModel } from "../../connect/mongo/model/order";

export default class {
    addModel(info: OrderModel) {
        return modelModel.insert(info);
    }
    removeModel(filter: FilterQuery<any>) {
        return modelModel.removeOne<any>(filter);
    }
    getModel(filter: FilterQuery<any>) {
        return modelModel.findOne<OrderModel>(filter);
    }
    updateModel(filter: FilterQuery<any>, update: UpdateQuery<any>) {
        return modelModel.updateOne<OrderModel>({ filter, update });
    }
}
