<template>
    <a-card class="count-data" :body-style="{ display: 'block' }">
        <div class="group">
            <span>统计字段:</span>
            <a-input class="input" placeholder="请输入需要被统计的字段" v-model:value="field" />
        </div>
        <div class="group">
            <span>截取长度:</span>
            <a-input-number class="input" placeholder="输入需要统计的组别的个数,默认全部输出" v-model:value="length" />
        </div>
        <div class="group">
            <a-upload v-model:file-list="fileList" :show-upload-list="false" :custom-request="upload">
                <a-button type="primary" style="margin: 0 20px">上传文件</a-button>
            </a-upload>
        </div>
    </a-card>
    <div v-if="getData">
        <a-card class="render-card" ref="line-chart" />
        <a-card class="render-card" ref="pie-chart" />
    </div>
</template>

<script setup>
import { getCurrentInstance, nextTick, ref } from "vue";
import * as echarts from "echarts";
import { message } from "ant-design-vue";
import { getColor, readData } from "../utils";

const field = ref("");
const fileList = ref([]);
const group = ref([]);
const length = ref();
const instance = getCurrentInstance();
const getData = ref(false);
const upload = async () => {
    if (!field.value) {
        message.warning("请输入excel表中需要被统计的字段");
        return;
    }
    getData.value = false;
    const file = fileList.value[0].originFileObj;
    const [data] = await readData(file);
    getData.value = true;
    groupByField(data);
};
const groupByField = data => {
    const key = field.value;
    const list = [];
    data.forEach(row => {
        const target = list.find(item => item.name === row[key]);
        if (target) {
            target.value += 1;
        } else {
            list.push({ value: 1, name: row[key] });
        }
    });
    group.value = list.sort((a, b) => b.value - a.value);
    nextTick(render);
};
const render = () => {
    const data = group.value.slice(0, length.value || group.value.length);
    const line = instance.refs["line-chart"].$el;
    const pie = instance.refs["pie-chart"].$el;
    echarts.init(line).setOption({
        xAxis: { type: "category", data: data.map(item => item.name) },
        toolbox: { feature: { saveAsImage: { title: "保存到本地" } } },
        tooltip: { trigger: "item", formatter: "{b} : {c}" },
        yAxis: { type: "value" },
        series: [
            {
                data: data.map(item => ({
                    value: item.value,
                    itemStyle: { color: getColor() },
                })),
                type: "bar",
            },
        ],
    });
    echarts.init(pie).setOption({
        tooltip: { trigger: "item" },
        toolbox: { feature: { saveAsImage: { title: "保存到本地" } } },
        legend: { type: "scroll", top: 20, bottom: 20, left: 20, orient: "vertical" },
        tooltip: { trigger: "item", formatter: "{a} <br/>{b} : {c} ({d}%)" },
        series: [
            {
                name: `${field.value}`,
                type: "pie",
                radius: "60%",
                data,
                emphasis: { itemStyle: { shadowBlur: 10, shadowOffsetX: 0, shadowColor: "rgba(0, 0, 0, 0.5)" } },
            },
        ],
    });
};
</script>

<style lang="less" scoped>
.count-data {
    margin: 16px;
    .group {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 8px;
        span {
            white-space: nowrap;
            padding: 0 8px;
        }
        .input {
            width: 300px;
        }
    }
}
.render-card {
    margin: 0 16px;
    margin-bottom: 16px;
    height: 500px;
}
</style>
