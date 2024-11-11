import mongoose, { FilterQuery, ProjectionType, SortOrder, UpdateQuery, QueryOptions } from "mongoose";

interface returnData<T> {
    data?: T;
    code?: "success" | "error";
    msg?: string;
    [x: string]: any;
}
class Mongo {
    model: mongoose.Model<any, unknown, unknown, unknown, any, any>;

    constructor({ schema, collection }) {
        this.model = mongoose.model(collection, schema);
    }

    async insert(info: any): Promise<returnData<any>> {
        const Model = this.model;
        const data = new Model(info);
        await data.save();
        return { ...data, code: "success", msg: "添加成功" };
    }

    async find<I>({
        filter = {},
        pageSize = 10,
        currentPage = 0,
        projection = {},
        sort = {},
    }: {
        filter?: FilterQuery<any>;
        pageSize?: number;
        currentPage?: number;
        projection?: ProjectionType<any>;
        sort?: { [key: string]: SortOrder | { $meta: any } };
    }): Promise<returnData<I>> {
        const dataPromise = this.model
            .find(filter, projection)
            .sort(sort)
            .limit(pageSize)
            .skip(currentPage * pageSize);
        const totalPromise = this.model.countDocuments(filter);
        const [data, total] = (await Promise.all([dataPromise, totalPromise])) as any;
        return {
            data,
            total,
            code: "success",
            msg: "查找成功",
        };
    }

    async findOne<I>(filter: FilterQuery<any> = {}): Promise<returnData<I>> {
        const data = await this.model.findOne(filter);
        return {
            data,
            msg: "查找成功",
            code: "success",
        };
    }

    async updateOne<I>({
        filter,
        update,
        options = { new: true },
    }: {
        filter: FilterQuery<any>;
        update: UpdateQuery<any>;
        options?: QueryOptions<any>;
    }): Promise<returnData<I>> {
        if (!filter || !update) throw new Error("请确保数据完整");
        const data = await this.model.findOneAndUpdate(filter, update, options);
        return {
            data,
            code: "success",
            msg: "修改成功",
        };
    }

    async removeMany<I>(filter: FilterQuery<any>): Promise<returnData<I>> {
        if (!filter) throw new Error("请确保数据完整");
        const data = (await this.model.deleteMany(filter)) as any;
        return { code: "success", msg: "删除成功", data };
    }

    async removeOne<I>(filter: FilterQuery<any>): Promise<returnData<I>> {
        if (!filter) throw new Error("请确保数据完整");
        const { deletedCount } = await this.model.deleteOne(filter);
        return deletedCount === 1
            ? { code: "success", msg: "删除成功" }
            : { code: "error", msg: "删除失败: 数据可能不存在" };
    }
    async aggregate<I>(...rest: any[]): Promise<returnData<I>> {
        const data = (await this.model.aggregate(...rest)) as any;
        return {
            msg: "查找成功",
            data,
            code: "success",
        };
    }
}

export const Model = (options: { schema: any; collection: any }) => new Mongo(options);

export const Schema = mongoose.Schema;
