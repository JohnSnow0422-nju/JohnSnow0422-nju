<template>
    <div class="tool-bar">
        <div style="display: flex; align-items: center">
            <img class="logo" src="../assets/logo.png" />
            <a-dropdown class="page-dropdown" :trigger="['click']">
                <a-button type="link" style="font-size: 16px">
                    {{ selectedKey?.label || "选择" }}
                    <down-outlined />
                </a-button>
                <template #overlay>
                    <a-menu :items="items" :selected-keys="selectedKeys" @click="to" />
                </template>
            </a-dropdown>
        </div>
        <div class="button-list">
            <a-button @click="openTour" type="primary" danger shape="circle" :icon="h(QuestionOutlined)" />
            <a-button type="primary" shape="circle" :icon="h(UserOutlined)" @click="to({ key: '/user-info' })" />
        </div>
    </div>
</template>

<script setup>
import { onMounted, ref, computed, h } from "vue";
import { DownOutlined, QuestionOutlined, UserOutlined } from "@ant-design/icons-vue";
import { useRouter } from "vue-router";
import { tourStore } from "../store/tour";

const router = useRouter();
const selectedKeys = ref(["/up-data"]);
const tour = tourStore();
const items = ref([
    { label: "上传", key: "/up-data" },
    { label: "下载", key: "/down-data" },
    { label: "统计", key: "/count-data" },
]);
const selectedKey = computed(() => items.value.find(item => item.key === selectedKeys.value?.[0]));
const to = ({ key }) => {
    tour.open = false;
    selectedKeys.value = [key];
    router.push({ path: key });
};
const openTour = () => (tour.open = true);

onMounted(() => {
    const href = new URL(window.location.href);
    const path = href.pathname;
    const key = items.value.find(item => item.key === path)?.key || "/up-data";
    selectedKeys.value = [key];
});
</script>

<style lang="less" scoped>
.tool-bar {
    background-color: white;
    padding: 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    .logo {
        height: 32px;
    }
}
.button-list {
    .ant-btn {
        margin-left: 8px;
    }
}
</style>
