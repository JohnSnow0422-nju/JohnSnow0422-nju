<template>
    <a-upload-dragger v-bind="options" v-model:file-list="data" :before-upload="() => false">
        <p class="ant-upload-drag-icon">
            <inbox-outlined />
        </p>
        <p class="ant-upload-text">点击或者拖动数据文件上传</p>
        <p class="ant-upload-hint">只支持上传xlsx文件,具体格式可点击右上角问号进行查看</p>
    </a-upload-dragger>
</template>

<script setup>
import { computed } from "vue";
import { InboxOutlined } from "@ant-design/icons-vue";

const props = defineProps({
    options: { type: Object, default: () => ({}) },
    value: { type: Array, default: [] },
});
const emit = defineEmits(["update:value"]);

const data = computed({
    get: () => props.value,
    set: val => emit("update:value", val),
});
</script>
