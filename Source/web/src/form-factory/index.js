import bert from "./bertopic";
const defaultConfig = [
    { component: "a-input", ragName: "area", options: { placeholder: "请输入数据所属领域,若不输入则由ai自动" } },
    {
        component: "a-input-number",
        ragName: "topicNum",
        options: { placeholder: "输入主题数,若不输入则自动取最优", style: "width: 100%", min: 1 },
    },
    {
        component: "a-input-number",
        ragName: "seed",
        options: { placeholder: "影响生成结果的参数,若不设置将随机生成", style: "width: 100%" },
    },
];
const map = {
    bert,
};
export default function Factory(model) {
    const config = [...defaultConfig];
    const replaceList = map[model] || [];
    replaceList.forEach(item => {
        const index = config.findIndex(i => i.ragName === item.ragName);
        if (index > -1) {
            config.splice(index, 1, item);
            return;
        }
        config.push(item);
    });
    return config;
}
