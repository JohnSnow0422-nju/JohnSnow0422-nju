<template>
    <a-card class="up-data" :body-style="{ justifyContent: 'center', flexDirection: 'column', alignItems: 'center' }">
        <a-steps style="width: 48%; margin-bottom: 16px" :current="step" :items="operationSteps"></a-steps>
        <d-form
            ref="form"
            :rules="rules"
            :model="formState"
            :config="formConfig[step]"
            v-model:value="formState"
            :wrapper-col="{ style: { width: '480px' } }" />
        <a-button style="width: 480px" type="primary" @click="buttonConfig[step].click" :loading="uploadLoading">
            {{ buttonConfig[step].label }}
        </a-button>
    </a-card>
    <a-tour
        :open="tour.open"
        :nextButtonProps="{ children: '自定义文字' }"
        :steps="tourSteps"
        @close="tour.open = false" />
</template>

<script setup>
import { computed, getCurrentInstance, onMounted, reactive, ref } from "vue";
import { message, notification } from "ant-design-vue";
import { readData } from "../utils";
import axios from "../apis/request";
import { tourStore } from "../store/tour";
import DForm from "../components/dynamic-form.vue";
import Uploader from "../components/uploader.vue";
import FormFactory from "../form-factory";

const instance = getCurrentInstance();
const formState = reactive({});
const tour = tourStore();
const operationSteps = [{ title: "必需项" }, { title: "可选项" }];
const rules = {
    model: [{ required: true, trigger: "change", message: "请选择数据对应合适的模型" }],
    fileList: [
        {
            required: true,
            trigger: "change",
            validator: async (_, fileList) => {
                const file = fileList?.[0]?.originFileObj;
                if (!file) return Promise.reject("请上传数据文件,用excel文件表示");
                const [fileData] = await readData(file);
                const isHasField = (needFieldList = [], data) =>
                    needFieldList.map(field => data.some(item => !item[field])).every(item => !item);
                const options = formConfig.value[0].find(i => i.ragName === "model").options.options;
                const target = options.find(item => item.value === formState?.model);
                if (!isHasField(target?.config?.field, fileData)) {
                    return Promise.reject(`${(target?.config?.field || [])?.join("、")}字段都是必须的`);
                }
                return Promise.resolve();
            },
        },
    ],
};
const formConfig = computed(() => [
    [
        {
            component: "a-select",
            ragName: "model",
            options: {
                ref: "model-ref",
                placeholder: "选择模型",
                options: [
                    { label: "lda主题模型", value: "lda", config: { field: ["content"], key: 1 } },
                    {
                        label: "dtm主题模型",
                        value: "dtm",
                        config: { field: ["content", "time"], sort: "time", key: 2 },
                    },
                    {
                        label: "bertopic主题模型",
                        value: "bert",
                        config: { field: ["content"], key: 3 },
                    },
                    { label: "atm主题模型", value: "atm", config: { field: ["author", "content"], key: 4 } },
                ],
            },
        },
        { component: Uploader, ragName: "fileList", options: { accept: ".xlsx", maxCount: 1 } },
    ],
    FormFactory(formState.model),
]);

const step = ref(0);
const tourSteps = computed(() => {
    return step.value === 0
        ? [
              {
                  title: "选择模型",
                  description:
                      "需要选择合适的模型, 不同的模型需要的表头字段也存在不同,且需要注意表头字段的大小写问题。",
                  target: () => instance.refs?.["model-ref"]?.$el,
              },
              {
                  title: "上传文件",
                  description:
                      "上传xlsx文件, 注意为excel工作簿格式, 而不是电子表格格式。同时excel文件中的表头需要统一为小写, 必需的表头不能存在空值。",
                  target: () => instance.refs?.["upload-ref"]?.$el,
              },
          ]
        : [
              {
                  title: "输入领域",
                  description: "输入数据所在领域, 让ai分析更精准。",
                  target: () => instance.refs?.["area-ref"]?.$el,
              },
              {
                  title: "输入主题数量",
                  description:
                      "输入主题数量, 确定最终生成的最大主题数量, 通常当使用者比较确定需要该主题数量后输入, 若第一次不确定主题数量多少合适, 可以有模型自己判断。",
                  target: () => instance.refs?.["topic-num-ref"]?.$el,
              },
              {
                  title: "输入随机种子",
                  description:
                      "输入随机种子, 不同的随机种子可能会使相同的实验数据产生不同的数据结果, 当获得了较为心仪的数据结果后尽量保持随机种子的值不变, 微调其他数据从而保证数据结果可复现。",
                  target: () => instance.refs?.["seed-ref"]?.$el,
              },
          ];
});

const uploadLoading = ref(false);
const buttonConfig = [
    {
        label: "下一步",
        click: async () => {
            try {
                const form = instance.refs.form;
                await form.validate();
                step.value += 1;
            } catch (error) {
                console.log(error);
                return;
            }
        },
    },
    {
        label: "上传",
        click: async () => {
            try {
                uploadLoading.value = true;
                const formData = new FormData();
                const file = formState?.fileList?.[0].originFileObj;
                const options = formConfig.value[0].find(i => i.ragName === "model").options.options;
                const config = options.find(item => item.value === formState?.model)?.config || {};
                const params = JSON.parse(JSON.stringify(formState));
                params.seed = formState.seed || parseInt(Math.random() * 10000);
                delete params.fileList;
                const uploadData = {
                    file,
                    params: JSON.stringify(params),
                    config: JSON.stringify(config),
                };
                Object.keys(uploadData).forEach(key => {
                    formData.append(key, uploadData[key]);
                });
                const {
                    data: { data, msg, code },
                } = await axios.post(`/model/info`, formData, { headers: { "Content-Type": "multipart/form-data" } });
                uploadLoading.value = false;
                if (code !== "success") {
                    message[code](msg);
                    return;
                }
                const localOrders = JSON.parse(localStorage.getItem("order-list") || "[]");
                localStorage.setItem("order-list", JSON.stringify([data, ...localOrders]));
                notification[code]({
                    placement: "topRight",
                    message: `本次消耗积分${msg}`,
                    description: `模型编号: ${data}`,
                });
                step.value = 0;
                Object.keys(formState).forEach(key => {
                    delete formState[key];
                });
            } catch (error) {
                message.error(`${error}`);
                uploadLoading.value = false;
            }
        },
    },
];
</script>

<style lang="less" scoped>
.up-data {
    display: flex;
    margin: 16px;
    overflow: auto;
    .form-data {
        display: flex;
        flex-direction: column;
    }
}
</style>
