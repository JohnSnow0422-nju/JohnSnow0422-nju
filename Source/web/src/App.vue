<template>
    <a-config-provider :locale="zhCN">
        <router-view />
    </a-config-provider>
</template>

<script setup>
import zhCN from "ant-design-vue/es/locale/zh_CN";
</script>

<style lang="less">
#app {
    background-color: #f2f3f5;
    height: 100%;
    width: 100%;
    overflow: auto;
}
</style>
