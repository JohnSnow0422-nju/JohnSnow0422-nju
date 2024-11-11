<template>
    <a-form ref="form">
        <a-form-item v-for="item in config" :name="item.ragName" :label="item.label">
            <component :is="item.component" v-bind="item.options" v-model:value="data[item.ragName]" />
        </a-form-item>
    </a-form>
</template>

<script setup>
import { computed, getCurrentInstance, ref } from "vue";

const props = defineProps({
    config: { type: Array, default: [] },
    value: { type: Object, default: () => ({}) },
});
const emit = defineEmits(["update:value"]);
const data = computed({
    get: () => props.value,
    set: val => emit("update:value", val),
});

const instance = getCurrentInstance();
const validate = async () => {
    await instance.refs.form.validate();
};

defineExpose({
    validate,
});
</script>

<style lang="less" scoped></style>
