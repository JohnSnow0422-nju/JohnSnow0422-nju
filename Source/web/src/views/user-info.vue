<template>
    <a-card class="user-page" :body-style="{ justifyContent: 'center', flexDirection: 'column', alignItems: 'center' }">
        <div v-if="userInfo.username">
            <a :href="userInfo.payLink" target="_blank">充值</a>
        </div>
        <a-form v-else ref="form" :rules="rules" :model="userform" :wrapper-col="{ style: { width: '400px' } }">
            <a-form-item name="username">
                <a-input v-model:value="userform.username" placeholder="输入邮箱,若邮箱未被注册则自动注册">
                    <template #addonBefore> <user-outlined /> </template>
                </a-input>
            </a-form-item>
            <a-form-item name="password">
                <a-input-password v-model:value="userform.password" placeholder="输入密码">
                    <template #addonBefore> <lock-outlined /> </template>
                </a-input-password>
            </a-form-item>
            <a-button type="primary" style="width: 100%" @click="login">登陆</a-button>
        </a-form>
    </a-card>
</template>

<script setup>
import { getCurrentInstance, reactive } from "vue";
import { LockOutlined, UserOutlined } from "@ant-design/icons-vue";
import { loginApi } from "../apis/pay";
import { userStore } from "../store/user";
import { message } from "ant-design-vue";

const rules = {
    username: [
        {
            trigger: "change",
            validator: async (_, data = "") => {
                if (!data) return Promise.reject("请输入邮箱!");
                const regex = /^([a-zA-Z0-9_\.-]+)@([\da-z\.-]+)\.([a-z\.]{2,6})$/;
                return regex.test(data) ? Promise.resolve() : Promise.reject("请输入正确的邮箱!");
            },
        },
    ],
    password: [{ trigger: "change", required: true, message: "请输入密码" }],
};
const instance = getCurrentInstance();
const userform = reactive({});
const userInfo = userStore();
const login = async () => {
    try {
        await instance.refs["form"].validate();
        const { code, msg } = await loginApi(userform);
        message[code](msg);
    } catch (error) {
        console.log(`error`);
        console.log(error);
        return;
    }
};
</script>

<style lang="less" scoped>
.user-page {
    margin: 16px;
}
</style>
