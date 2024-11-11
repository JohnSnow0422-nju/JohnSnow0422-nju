<template>
    <a-config-provider :theme="{ token: { colorPrimary: '#2538C4' } }">
        <tool-bar />
        <router-view />
    </a-config-provider>
</template>

<script setup>
import ToolBar from "../components/tool-bar.vue";
</script>

<style lang="less">
.ant-card-body {
    width: 100%;
    display: flex;
}
</style>
