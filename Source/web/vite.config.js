import { fileURLToPath, URL } from "node:url";
import path from "node:path";
import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import { createHash } from "node:crypto";

// https://vitejs.dev/config/
export default defineConfig({
    plugins: [vue()],
    // root: path.resolve(__dirname, "./src"),
    build: {
        outDir: path.resolve(__dirname, "./dist"),
        rollupOptions: {
            input: { 1: path.resolve(__dirname, "./index.html") },
            output: {
                entryFileNames: `[name].[hash].js`,
                chunkFileNames: `[name].[hash].js`,
                assetFileNames: `[name].[hash].[ext]`,
                manualChunks: id =>
                    id.includes("node_modules") &&
                    createHash("md5")
                        .update(id.toString().split("node_modules/")[1].split("/")[0].toString())
                        .digest("hex"),
            },
        },
    },
    resolve: { alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) } },
    server: {
        host: "0.0.0.0",
        proxy: {
            "/model/": {
                target: "http://127.0.0.1:3000",
                changeOrigin: true,
            },
            "/pay/": {
                target: "http://127.0.0.1:3000",
                changeOrigin: true,
            },
            "/user/": {
                target: "http://127.0.0.1:3000",
                changeOrigin: true,
            },
        },
    },
});
