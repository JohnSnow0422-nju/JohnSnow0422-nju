import { createRouter, createWebHistory } from "vue-router";

// const modules = import.meta.glob("../apps/*/index.js", { eager: true });
const routes = [
    {
        path: "/",
        redirect: "/up-data",
        component: () => import("../views/home.vue"),
        home: true,
        children: [
            { path: "up-data", component: () => import("../views/up-data.vue") },
            { path: "down-data", component: () => import("../views/down-data.vue") },
            { path: "count-data", component: () => import("../views/count-data.vue") },
            { path: "user-info", component: () => import("../views/user-info.vue") },
        ],
    },
];

// window.apps = [];
// Object.keys(modules).forEach(key => {
//     const module = modules[key].default;
//     const routeList = module.router;
//     const path = key.split("/");
//     const appname = path[path.indexOf("apps") + 1];
//     if (Array.isArray(routeList)) {
//         const moduleRoutes = routeList.map(route => {
//             const path = `/${appname}${route.path}`;
//             if (route.home) {
//                 const data = JSON.parse(JSON.stringify(module));
//                 delete data.router;
//                 window.apps.push({ ...data, path });
//             }
//             if (route.redirect) {
//                 route.redirect = `/${appname}${route.redirect}`;
//             }
//             return { ...route, path };
//         });
//         routes.push(...moduleRoutes);
//     }
// });

// console.log(routes);
const router = createRouter({
    history: createWebHistory(import.meta.env.BASE_URL),
    routes,
});

export default router;
