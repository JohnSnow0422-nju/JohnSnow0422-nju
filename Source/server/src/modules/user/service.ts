import { FilterQuery } from "mongoose";
import userModel, { UserModel } from "../../connect/mongo/model/user";

export default class {
    getByUsername(filter: FilterQuery<any>) {
        return userModel.findOne<UserModel>(filter);
    }
    add(user: UserModel) {
        return userModel.insert(user);
    }
}
