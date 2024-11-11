import mongoose from "mongoose";
import { createClient, RedisClientType } from "redis";
import { mongoUri, redisUri } from "./config";
export const redis: RedisClientType = createClient();

export async function mongoConnect() {
    await mongoose.connect(mongoUri);
    console.log("connect to mongo successful");
}

export async function redisConnect() {
    await redis.connect();
    console.log("connect to redis successful");
}

export async function mongoDisconnect() {
    await mongoose.connection.close();
    console.log("mongo unlink");
}
