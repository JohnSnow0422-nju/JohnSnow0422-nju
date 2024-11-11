<template>
    <a-card class="down-data" :body-style="{ justifyContent: 'center', flexDirection: 'column', alignItems: 'center' }">
        <div style="width: 480px; display: flex">
            <a-input
                style="flex: 1"
                v-model:value="order"
                placeholder="输入模型编号"
                @keyup.enter.native="onSearch"
                allow-clear />
            <a-button type="primary" style="margin-left: 16px" @click="onSearch">下载</a-button>
        </div>
        <div class="cache">
            <a-tag
                :key="tag"
                v-for="tag in ordersCache"
                closable
                style="cursor: pointer"
                @close="removeCache(tag)"
                @click="order = tag"
                :color="getColor()">
                {{ tag }}
            </a-tag>
        </div>
    </a-card>
    <div class="render-part" v-if="getData">
        <a-card class="download-file" title="可下载文件">
            <a-button type="link" v-for="filename in downloadFileList" @click="downloadFile(filename)">
                {{ filename }}
            </a-button>
        </a-card>
        <div class="render-fall">
            <div v-for="i in [0, 1]" :class="{ 'render-part-left': i === 0, 'render-part-right': i === 1 }">
                <div class="render-card" v-for="item in data.filter((_, index) => index % 2 === i)">
                    <div class="title">
                        <span class="title-text">{{ item.title }}</span>
                        <div class="button-list">
                            <a-button
                                :disabled="isLoading || !themeText"
                                title="数据分析"
                                size="small"
                                type="text"
                                @click="analysis(item)">
                                <template #icon>
                                    <sync-outlined v-if="isLoading" />
                                    <global-outlined v-else />
                                </template>
                            </a-button>
                        </div>
                    </div>
                    <div class="render-content" :ref="`${item.key}`" :style="{ height: `${item.height}px` }" />
                    <div
                        class="ai-analysis"
                        v-if="response[item.key]"
                        v-html="response[item.key].replace(/\r?\n/g, '<br/>')"></div>
                </div>
            </div>
        </div>
    </div>
</template>

<script setup>
import { ref, nextTick, getCurrentInstance, h, computed, reactive } from "vue";
import { message } from "ant-design-vue";
import axios from "../apis/request";
import { GlobalOutlined, SyncOutlined } from "@ant-design/icons-vue";
import { modelMap, askGpt, getColor } from "../utils";

const order = ref();
const getData = ref(false);
const data = ref([]);
const model = ref("");
const instance = getCurrentInstance();
const themeText = ref("");
const response = reactive({});
const onSearch = async () => {
    if (!order.value) return;
    themeText.value = "";
    getData.value = false;
    Object.keys(response).forEach(key => (response[key] = ""));
    getDownloadFileList();
    const { data: res } = await axios.get(`/model/report?id=${order.value}`);
    data.value = res.data;
    model.value = res.model;
    if (res.code === "error") {
        message.error(res.msg);
        return;
    }
    getData.value = true;
    nextTick(() => modelMap[model.value].render({ refs: instance.refs, data: data.value }));
    modelMap?.[model.value]?.getTheme &&
        (themeText.value = await modelMap[model.value].getTheme(data.value, order.value));
};
const reOrder = ref(true);
const removeCache = value => {
    const newCache = ordersCache.value.filter(i => i !== value);
    console.log(ordersCache.value, value, newCache);
    reOrder.value = false;
    localStorage.setItem("order-list", JSON.stringify(newCache));
    reOrder.value = true;
};
const ordersCache = computed(() => reOrder.value && JSON.parse(localStorage.getItem("order-list") || "[]"));

const isLoading = ref(false);
const analysis = async ({ key }) => {
    isLoading.value = true;
    const target = data.value.find(item => item.key === key);
    response[key] = localStorage.getItem(`${order.value}-${model.value}-${key}`) || "";
    if (!response?.[key]) {
        const prompt = `${themeText.value}\n${modelMap[model.value].prompt[key](target.data, data.value)}`;
        const gptResponse = await askGpt(prompt, response, key);
        gptResponse && localStorage.setItem(`${order.value}-${model.value}-${key}`, gptResponse);
    }
    isLoading.value = false;
};

const downloadFileList = ref([]);
const getDownloadFileList = async () => {
    const { data: res } = await axios.get(`/model/fileList?id=${order.value}`);
    downloadFileList.value = res.data;
};
const downloadFile = async filename => {
    const { data: res } = await axios.get(
        "/model/file",
        { params: { id: order.value, filename } },
        { responseType: "blob" }
    );
    const blob = new Blob([res]);
    const downloadElement = document.createElement("a");
    const href = window.URL.createObjectURL(blob);
    downloadElement.href = href;
    downloadElement.download = filename;
    document.body.appendChild(downloadElement);
    downloadElement.click();
    document.body.removeChild(downloadElement);
    window.URL.revokeObjectURL(href);
};
</script>

<style lang="less" scoped>
.down-data {
    margin: 16px;
    .cache {
        margin-top: 8px;
        width: 480px;
        white-space: nowrap;
        overflow: auto;
    }
}
.render-part {
    padding: 0 16px;
    .download-file {
        margin-bottom: 16px;
        overflow: auto;
    }
    .render-fall {
        display: flex;
        width: 100%;
        justify-content: space-between;
        .render-part-left,
        .render-part-right {
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        .render-part-left {
            margin-right: 16px;
        }
        .render-card {
            background-color: white;
            overflow: auto;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 16px;
            transition: all 0.2s;
            .ai-analysis {
                margin-top: 16px;
                font-size: 13px;
                color: #4e4e4e;
                transition: all 0.2s;
            }
            .render-content {
                width: 100%;
                height: 450px;
            }
            .title {
                margin-bottom: 5px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                .title-text {
                    font-size: 18px;
                }
                .button-list {
                    display: flex;
                    align-items: center;
                }
            }
        }
    }
}
</style>
