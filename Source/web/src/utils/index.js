import * as XLSX from "xlsx";
import * as echarts from "echarts";
// import { gpt } from "gpti";
import axios from "../apis/request";

const lineStyle = ["circle", "rect", "triangle", "diamond", "pin", "arrow", "roundRect", "none"];

function readData(file) {
    return new Promise(resolve => {
        const fileReader = new FileReader();
        fileReader.onload = e => {
            const result = e.target?.result;
            const workbook = XLSX.read(result, { type: "binary" });
            const res = workbook.SheetNames.map(sheet => XLSX.utils.sheet_to_json(workbook.Sheets[sheet]));
            resolve(res);
        };
        fileReader.readAsBinaryString(file);
    });
}

async function getModelInfo(id) {
    return axios.get(`/model/info?id=${id}`);
}

const modelMap = {
    dtm: {
        getTheme: async (map, _id) => {
            const {
                data: { data },
            } = await getModelInfo(_id);
            if (data?.themeText) return data.themeText;
            const area = data?.area ? `接下来的数据是属于${data?.area}领域的。` : "";
            const themeWords = ((map.find(item => item.key === "topicWordTree")?.data || [])[0]?.children || [])
                .map(
                    item =>
                        `${item.name}的主题词及分布概率为:\n${item.children
                            .map(word => `${word.name}:${word.value}`)
                            .join(",")}`
                )
                .join(";\n");
            const themeText = await askGpt(
                `${area}\n${themeWords},请你利用上述各主题词的主题词分布概率来判断各个主题可能是哪方面的,并且尽量用一个简洁的词汇或者短语来概括每个主题。`
            );
            axios.put("/model/info", { filter: { _id }, update: { themeText } });
            return themeText;
        },
        render: ({ refs, data }) => {
            const generateOption = {
                coherence: data => ({
                    tooltip: { trigger: "axis" },
                    legend: { data: (data?.series || []).map(item => item.name) },
                    grid: { left: "3%", right: "4%", bottom: "3%", containLabel: true },
                    toolbox: { feature: { saveAsImage: { title: "保存到本地" } } },
                    xAxis: { type: "category", boundaryGap: false, data: data?.xAxis || [] },
                    yAxis: { type: "value", min: data?.yMin },
                    series: (data?.series || []).map((item, index) => ({
                        ...item,
                        type: "line",
                        smooth: true,
                        symbolSize: 10,
                        symbol: lineStyle[index],
                    })),
                }),
                theme: data => ({
                    toolbox: { feature: { saveAsImage: { title: "保存到本地" } } },
                    tooltip: { trigger: "axis" },
                    legend: { data: (data?.series || []).map(item => item.name) },
                    grid: { left: "3%", right: "4%", bottom: "3%", containLabel: true },
                    xAxis: { type: "category", boundaryGap: false, data: data?.xAxis || [] },
                    yAxis: { type: "value", min: data?.yMin },
                    series: (data?.series || []).map((item, index) => ({
                        ...item,
                        type: "line",
                        smooth: true,
                        symbolSize: 10,
                        symbol: lineStyle[index],
                    })),
                }),
                sentiment: data => {
                    let min = 0,
                        max = 0;
                    (data?.data || []).forEach(item => {
                        min = Math.min(min, item?.[2] || 0);
                        max = Math.max(max, item?.[2] || 0);
                    });
                    return {
                        tooltip: { position: "top" },
                        toolbox: { feature: { saveAsImage: { title: "保存到本地" } } },
                        xAxis: { type: "category", data: data.xAxis, splitArea: { show: true } },
                        yAxis: { type: "category", data: data.yAxis, splitArea: { show: true } },
                        visualMap: { max, min, calculable: true, orient: "horizontal", left: "center" },
                        series: [
                            {
                                name: "平均情感值",
                                type: "heatmap",
                                data: data.data,
                                label: { show: true },
                                emphasis: { itemStyle: { shadowBlur: 10, shadowColor: "rgba(0, 0, 0, 0.5)" } },
                            },
                        ],
                    };
                },
                topicWordTree: data => ({
                    tooltip: { trigger: "item" },
                    toolbox: { feature: { saveAsImage: { title: "保存到本地" } } },
                    series: {
                        type: "sunburst",
                        data: data,
                        radius: [0, "95%"],
                        emphasis: { focus: "ancestor" },
                        levels: [
                            {},
                            { r0: "15%", r: "35%", itemStyle: { borderWidth: 2 }, label: { rotate: "tangential" } },
                            { r0: "35%", r: "80%", label: { align: "right" } },
                            {
                                r0: "80%",
                                r: "84%",
                                label: { show: Math.random() < 0.5, position: "outside", padding: 1, silent: false },
                                itemStyle: { borderWidth: 3 },
                            },
                        ],
                    },
                }),
                wordFrequency: data => ({
                    grid: { left: "3%", right: "4%", bottom: "3%", top: "3%", containLabel: true },
                    toolbox: { feature: { saveAsImage: { title: "保存到本地" } } },
                    tooltip: {
                        formatter: info =>
                            `<div class="tooltip-title">${info.data.name}的词频: ${info.data.value || "-"}</div>`,
                    },
                    series: [
                        {
                            name: "各年top10词频统计",
                            type: "treemap",
                            label: { show: true },
                            upperLabel: {
                                show: true,
                                height: 30,
                            },
                            itemStyle: { borderColor: "#fff" },
                            levels: [
                                {
                                    itemStyle: { borderColor: "#777", borderWidth: 0, gapWidth: 1 },
                                    upperLabel: { show: false },
                                },
                                {
                                    itemStyle: { borderColor: "#555", borderWidth: 5, gapWidth: 1 },
                                    emphasis: { itemStyle: { borderColor: "#ddd" } },
                                },
                                {
                                    colorSaturation: [0.35, 0.5],
                                    itemStyle: { borderWidth: 5, gapWidth: 1, borderColorSaturation: 0.6 },
                                },
                            ],
                            data: data,
                        },
                    ],
                }),
            };
            data.forEach(item => {
                const [ref] = refs[`${item.key}`];
                const myChart = echarts.init(ref);
                const option = generateOption?.[item.key] && generateOption?.[item.key](item.data);
                myChart.hideLoading();
                option && myChart.setOption(option);
            });
        },
        prompt: {
            theme: data => {
                const timeList = data?.xAxis;
                const prompt = (data.series || [])
                    .map(item => {
                        let str = "";
                        const theme = item.name;
                        const strong = item.data;
                        for (let i = 0; i < timeList.length; i++) {
                            str += `${theme}在时间${timeList[i]}的主题强度为:${strong[i]};`;
                        }
                        return `${str}\n`;
                    })
                    .join("\n");
                return `${prompt}根据上述的主题类型以及主题强度变化相关数据来描述主题强度变化趋势并且详细分析发生这种趋势变化可能存在的原因。`;
            },
            wordFrequency: data => {
                const prompt = (data || [])
                    .map(item => {
                        const time = item.name;
                        const str = (item?.children || []).map(word => `${word.name}的词频数为${word.value}`).join(",");
                        return `在时间${time}中,${str};\n`;
                    })
                    .join("\n");
                return `${prompt}根据上述的主题类型以及不同时间段的词频数相关数据描述不同时间段的词频数并且详细分析产生热门词汇的原因。`;
            },
            sentiment: data => {
                const sentimentList = data?.data || [];
                const timeList = data.xAxis;
                const themeList = data.yAxis;
                const prompt = themeList
                    .map((theme, index) => {
                        const filter = sentimentList
                            .filter(item => item?.[1] === index)
                            .map(item => `在时间${timeList[item[0]]}的平均情感值为${item[2]}`)
                            .join(",");
                        return `${theme}${filter};\n`;
                    })
                    .join("\n");
                return `${prompt}根据上述的主题类型以及不同时间段的平均主题情感值相关数据描述不同时间段的各主题的情感值并且详细分析情感变化的原因。(正数表示积极情绪，负数表示消极情绪，绝对值的大小表示情感强度)`;
            },
            coherence: data => {
                const time = data?.xAxis;
                const list = data.series.map(item => ({
                    str: item.name,
                    ave: (item?.data || []).reduce((a, b) => a + b) / (time?.length || 1),
                }));
                let max = 0,
                    maxTopic = "";
                let best = 0,
                    bestTopic = "";
                list.forEach(item => {
                    if (item.ave > max) {
                        max = item.ave;
                        maxTopic = item.str;
                    }
                });
                for (let i = 0; i < list.length; i++) {
                    if (list[i].ave > best) {
                        best = list[i].ave;
                        bestTopic = list[i].str;
                    } else {
                        break;
                    }
                }
                return `跑主题一致性时${bestTopic}的平均一致性为${best},${maxTopic}的平均一致性为${max},详细讲述一下为什么选择主题数更偏向选择${bestTopic}`;
            },
            topicWordTree: data => {
                const prompt = data
                    .map(item => {
                        const time = item.name;
                        const str = (item?.children || [])
                            .map(themeClass => {
                                const theme = themeClass.name;
                                const word = themeClass.children
                                    .slice(0, 5)
                                    .map(word => `${word.name}的概率:${word.value}`)
                                    .join(",");
                                return `${theme}中${word};`;
                            })
                            .join("\n");
                        return `在时间${time}中\n${str}`;
                    })
                    .join("。\n");
                return `${prompt}根据上述提供的数据来重述各个主题的主题类型并且对各个主题进行相关性分析,同时详细分析这些数据的数据价值并从这些方面对其进行分析。`;
            },
        },
    },
    lda: {
        getTheme: async (map, _id) => {
            const {
                data: { data },
            } = await getModelInfo(_id);
            if (data?.themeText) return data.themeText;
            const area = data?.area ? `接下来的数据是属于${data?.area}领域的。` : "";
            const themeWords = (map.find(item => item.key === "topicWordTree")?.data || [])
                .map(
                    item =>
                        `${item.name}的主题词及分布概率为${item.children
                            .map(word => `${word.name}:${word.value}`)
                            .join(",")}`
                )
                .join(";\n");

            const themeText = await askGpt(
                `${area}\n${themeWords},请你利用上述各主题词的主题词分布概率来判断各个主题可能是哪方面的,并且尽量用简洁的词汇或者短语来概括每个主题。`
            );
            axios.put("/model/info", { filter: { _id }, update: { themeText } });
            return themeText;
        },
        render: ({ refs, data }) => {
            const generateOption = {
                coherence: data => ({
                    tooltip: { trigger: "axis" },
                    xAxis: { data: data.xAxis },
                    toolbox: { feature: { saveAsImage: { title: "保存到本地" } } },
                    yAxis: {
                        type: "value",
                        min: (Math.min(...(data?.series?.[0]?.data || []).map(i => Number(i))) * 0.98).toFixed(2),
                    },
                    series: data.series,
                }),
                theme: data => ({
                    tooltip: { trigger: "item" },
                    color: ["#67F9D8", "#FFE434", "#56A3F1", "#FF917C"],
                    legend: { data: data.series.map(item => item.name) },
                    toolbox: { feature: { saveAsImage: { title: "保存到本地" } } },
                    radar: {
                        indicator: data.indicator,
                        axisName: { formatter: "【{value}】", color: "#428BD4" },
                        shape: "circle",
                        splitArea: {
                            areaStyle: {
                                color: ["#77EADF", "#26C3BE", "#64AFE9", "#428BD4"],
                                shadowColor: "rgba(0, 0, 0, 0.2)",
                                shadowBlur: 10,
                            },
                        },
                    },
                    series: [
                        {
                            type: "radar",
                            data: data.series.map(i => ({
                                ...i,
                                areaStyle: { color: "rgba(255, 228, 52, 0.9)" },
                            })),
                        },
                    ],
                }),
                wordFrequency: data => ({
                    grid: { left: "3%", right: "4%", bottom: "3%", top: "3%", containLabel: true },
                    tooltip: { formatter: "{b} : {c}次" },
                    toolbox: { feature: { saveAsImage: { title: "保存到本地" } } },
                    series: [
                        {
                            name: "top50词频统计",
                            type: "treemap",
                            label: { show: true },
                            upperLabel: { show: true, height: 30 },
                            itemStyle: { borderColor: "#fff" },
                            gap: 2,
                            label: { show: true, position: "inside" },
                            data: data.series,
                        },
                    ],
                }),
                topicWordTree: data => ({
                    tooltip: { trigger: "item", triggerOn: "mousemove" },
                    toolbox: { feature: { saveAsImage: { title: "保存到本地" } } },
                    series: [
                        {
                            type: "tree",
                            data: [{ children: data, name: "主题词分布概率" }],
                            layout: "radial",
                            animationDurationUpdate: 750,
                            emphasis: { focus: "ancestor" },
                        },
                    ],
                }),
                sentiment: data => {
                    const num = data.themeNum.length;
                    let max = 0;
                    for (let index = 0; index < (data?.data || []).length; index++) {
                        const target = (data?.data || [])?.[index];
                        max = Math.max(target?.[2], max);
                    }
                    return {
                        tooltip: { trigger: "axis" },
                        toolbox: { feature: { saveAsImage: { title: "保存到本地" } } },
                        title: data.themeNum.map((item, index) => ({
                            textBaseline: "middle",
                            top: ((index + 0.5) * 100) / num + "%",
                            text: `主题${item}`,
                        })),
                        singleAxis: data.themeNum.map((_, index) => ({
                            left: 150,
                            type: "category",
                            boundaryGap: false,
                            top: (index * 100) / num + 5 + "%",
                            height: 100 / num - 10 + "%",
                        })),
                        series: data.themeNum.map((_, index) => {
                            const filter = data.data.filter(item => item[0] === _).sort((a, b) => a[1] - b[1]);
                            return {
                                singleAxisIndex: index,
                                data: filter.map(i => [i[1], i[2]]),
                                coordinateSystem: "singleAxis",
                                type: "scatter",
                                symbolSize: i => parseInt(Math.min((i[1] / max) * 16, 100)),
                            };
                        }),
                    };
                },
            };
            data.forEach(item => {
                const [ref] = refs[`${item.key}`];
                const myChart = echarts.init(ref);
                const option = generateOption?.[item.key] && generateOption?.[item.key](item.data);
                myChart.hideLoading();
                option && myChart.setOption(option);
            });
        },
        prompt: {
            coherence: data => {
                const coherenceValueList = data.series?.[0]?.data || [];
                let max = 0,
                    maxTopic = "",
                    best = 0,
                    bestTopic = "";
                for (let i = 0; i < coherenceValueList.length; i++) {
                    if (coherenceValueList[i] > max) {
                        max = coherenceValueList[i];
                        maxTopic = data.xAxis[i];
                    }
                }
                for (let i = 0; i < coherenceValueList.length; i++) {
                    if (coherenceValueList[i] > best) {
                        best = coherenceValueList[i];
                        bestTopic = data.xAxis[i];
                    } else {
                        break;
                    }
                }
                return `跑主题一致性时${bestTopic}的平均一致性为${best},${maxTopic}的平均一致性为${max},详细讲述一下为什么选择主题数更偏向选择${bestTopic}`;
            },
            theme: data => {
                const prompt = data.indicator
                    .map((item, index) => `${item.name}的主题强度为${data.series?.[0]?.value?.[index]}`)
                    .join(",");
                return `${prompt}。\n根据上述的主题类型以及主题强度相关数据来描述各个主题的主题强度并且详细分析发生这种现象可能存在的原因。`;
            },
            wordFrequency: data => {
                const prompt = (data.series || []).map(item => `${item.name}的词频数为${item.value}`).join(",");
                return `${prompt}。\n根据上述的主题类型以及词频数相关数据描述词频数并且详细分析产生热门词汇的原因。`;
            },
            topicWordTree: data => {
                const prompt = data
                    .map(
                        topic =>
                            `${topic.name}中${(topic?.children || [])
                                .map(word => `${word.name}的概率为${word.value}`)
                                .join(",")}`
                    )
                    .join("。\n");
                return `${prompt}。\n根据上述提供的数据来重述各个主题的主题类型并且对各个主题进行相关性分析,同时详细分析这些数据的数据价值并从这些方面对其进行分析。`;
            },
            sentiment: data => {
                const prompt = (data?.themeNum || [])
                    .map(topicIndex => {
                        const filter = (data?.data || [])
                            .filter(item => item[0] === topicIndex)
                            .map(item => `情感值为${item[1]}的文本数为${item[2]}`)
                            .join(",");
                        return `主题${topicIndex}中,${filter}。`;
                    })
                    .join("\n");
                return `${prompt}。\n根据上述的主题类型以及各主题中不同情感值的文本数来进行分析其中可能的原因。(正数表示积极情绪,负数表示消极情绪,绝对值的大小表示情感的强度)`;
            },
        },
    },
    bert: {
        getTheme: async (map, _id) => {
            const {
                data: { data },
            } = await getModelInfo(_id);
            if (data?.themeText) return data.themeText;
            const area = data?.area ? `接下来的数据是属于${data?.area}领域的。` : "";
            const themeWords = (map.find(item => item.key === "topicWordTree")?.data || [])
                .map(
                    item =>
                        `${item.name}的主题词及分布概率为${item.children
                            .map(word => `${word.name}:${word.value}`)
                            .join(",")}`
                )
                .join(";\n");
            const themeText = await askGpt(
                `${area}\n${themeWords},但需要注意的是，主题-1是没有分配主题的离群值。请你利用上述各主题词的主题词分布概率来判断除了主题-1外的各个主题可能是哪方面的,并且尽量用简洁的词汇或者短语来概括每个主题。并说明主题-1存在的可能性。`
            );
            axios.put("/model/info", { filter: { _id }, update: { themeText } });
            return themeText;
        },
        render: ({ refs, data }) => {
            const generateOption = {
                sentiment: data => {
                    const num = data.themeNum.length;
                    let max = 0;
                    for (let index = 0; index < (data?.data || []).length; index++) {
                        const target = (data?.data || [])?.[index];
                        // console.log(target?.[2]);
                        max = Math.max(target?.[2], max);
                    }
                    return {
                        tooltip: { trigger: "axis" },
                        toolbox: { feature: { saveAsImage: { title: "保存到本地" } } },
                        title: data.themeNum.map((item, index) => ({
                            textBaseline: "middle",
                            top: ((index + 0.5) * 100) / num + "%",
                            text: `主题${item}`,
                        })),
                        singleAxis: data.themeNum.map((_, index) => ({
                            left: 150,
                            type: "category",
                            boundaryGap: false,
                            top: (index * 100) / num + "%",
                            height: 100 / num - 10 + "%",
                        })),
                        series: data.themeNum.map((_, index) => {
                            const filter = data.data.filter(item => item[0] === _).sort((a, b) => a[1] - b[1]);
                            return {
                                singleAxisIndex: index,
                                data: filter.map(i => [i[1], i[2]]),
                                coordinateSystem: "singleAxis",
                                type: "scatter",
                                symbolSize: i => parseInt(Math.min((i[1] / max) * 16, 100)),
                            };
                        }),
                    };
                },
                topicWordTree: data => ({
                    tooltip: { trigger: "item", triggerOn: "mousemove" },
                    toolbox: { feature: { saveAsImage: { title: "保存到本地" } } },
                    series: [
                        {
                            type: "tree",
                            data: [{ children: data, name: "主题词分布概率" }],
                            layout: "radial",
                            animationDurationUpdate: 750,
                            emphasis: { focus: "ancestor" },
                        },
                    ],
                }),
                wordFrequency: data => ({
                    grid: { left: "3%", right: "4%", bottom: "3%", top: "3%", containLabel: true },
                    tooltip: { formatter: "{b} : {c}次" },
                    toolbox: { feature: { saveAsImage: { title: "保存到本地" } } },
                    series: [
                        {
                            name: "top50词频统计",
                            type: "treemap",
                            label: { show: true },
                            upperLabel: { show: true, height: 30 },
                            itemStyle: { borderColor: "#fff" },
                            gap: 2,
                            label: { show: true, position: "inside" },
                            data: data.series,
                        },
                    ],
                }),
                theme: data => ({
                    tooltip: { trigger: "axis" },
                    toolbox: { feature: { saveAsImage: { title: "保存到本地" } } },
                    xAxis: { type: "category", data: data.xAxis },
                    yAxis: { type: "value" },
                    series: [
                        {
                            data: (data?.series?.data || []).map(item => ({
                                value: item,
                                itemStyle: { color: getColor() },
                            })),
                            type: "bar",
                        },
                    ],
                }),
            };
            data.forEach(item => {
                const [ref] = refs[`${item.key}`];
                const myChart = echarts.init(ref);
                const option = generateOption?.[item.key] && generateOption?.[item.key](item.data);
                myChart.hideLoading();
                option && myChart.setOption(option);
            });
        },
        prompt: {
            theme: data => {
                const prompt = (data?.xAxis || [])
                    .map((topic, i) => `${topic}的相关文档数量为${data?.series?.data?.[i] || 0}`)
                    .join(",");
                return `${prompt}。\n根据上述的主题类型以及主题文档数相关数据来描述各个主题的主题强度并且详细分析发生这种现象可能存在的原因。`;
            },
            wordFrequency: data => {
                const prompt = (data.series || []).map(item => `${item.name}的词频数为${item.value}`).join(",");
                return `${prompt}。\n根据上述的主题类型以及词频数相关数据描述词频数并且详细分析产生热门词汇的原因。`;
            },
            topicWordTree: data => {
                const prompt = data
                    .map(
                        topic =>
                            `${topic.name}中${(topic?.children || [])
                                .map(word => `${word.name}的概率为${word.value}`)
                                .join(",")}`
                    )
                    .join("。\n");
                return `${prompt}。\n根据上述提供的数据来重述各个主题的主题类型并且对各个主题进行相关性分析,同时详细分析这些数据的数据价值并从这些方面对其进行分析。`;
            },
            sentiment: data => {
                const prompt = (data?.themeNum || [])
                    .map(topicIndex => {
                        const filter = (data?.data || [])
                            .filter(item => item[0] === topicIndex)
                            .map(item => `情感值为${item[1]}的文本数为${item[2]}`)
                            .join(",");
                        return `主题${topicIndex}中,${filter}。`;
                    })
                    .join("\n");
                return `${prompt}。\n根据上述的主题类型以及各主题中不同情感值的文本数来进行分析其中可能的原因。(正数表示积极情绪,负数表示消极情绪,绝对值的大小表示情绪的强度)`;
            },
        },
    },
    atm: {
        getTheme: async (map, _id) => {
            const {
                data: { data },
            } = await getModelInfo(_id);
            if (data?.themeText) return data.themeText;
            const area = data?.area ? `接下来的数据是属于${data?.area}领域的。` : "";
            const themeWords = (map.find(item => item.key === "topicWordTree")?.data || [])
                .map(
                    item =>
                        `${item.name}的主题词及分布概率为${item.children
                            .map(word => `${word.name}:${word.value}`)
                            .join(",")}`
                )
                .join(";\n");
            const themeText = await askGpt(
                `${area}\n${themeWords},请你利用上述各主题词的主题词分布概率来判断各个主题可能是哪方面的,并且尽量用简洁的词汇或者短语来概括每个主题。`
            );
            axios.put("/model/info", { filter: { _id }, update: { themeText } });
            return themeText;
        },
        render: ({ refs, data }) => {
            const generateOption = {
                coherence: data => ({
                    tooltip: { trigger: "axis" },
                    xAxis: { data: data.xAxis },
                    toolbox: { feature: { saveAsImage: { title: "保存到本地" } } },
                    yAxis: {
                        type: "value",
                        min: (Math.min(...(data?.series?.[0]?.data || []).map(i => Number(i))) * 0.98).toFixed(2),
                    },
                    series: data.series,
                }),
                publishData: data => ({
                    tooltip: { trigger: "item" },
                    toolbox: { feature: { saveAsImage: { title: "保存到本地" } } },
                    legend: { orient: "vertical", left: "right", type: "scroll" },
                    series: data.series,
                }),
                topicWordTree: data => ({
                    tooltip: { trigger: "item", triggerOn: "mousemove" },
                    toolbox: { feature: { saveAsImage: { title: "保存到本地" } } },
                    series: [
                        {
                            type: "tree",
                            data: [{ children: data, name: "主题词分布概率" }],
                            layout: "radial",
                            animationDurationUpdate: 750,
                            emphasis: { focus: "ancestor" },
                        },
                    ],
                }),
                authorTopicRelation: data => ({
                    tooltip: { trigger: "item", triggerOn: "mousemove" },
                    toolbox: { feature: { saveAsImage: { title: "保存到本地" } } },
                    legend: [{ data: data.categories.map(i => i.name) }],
                    series: [
                        {
                            type: "graph",
                            layout: "force",
                            data: data.nodes,
                            links: data.links,
                            roam: true,
                            draggable: true,
                            label: { show: true, position: "right" },
                            labelLayout: { hideOverlap: true },
                            emphasis: { focus: "adjacency", label: { position: "right", show: true } },
                            categories: data.categories.map(i => ({ name: i.name })),
                            force: {
                                repulsion: 12,
                                edgeLength: 50,
                            },
                        },
                    ],
                }),
            };
            data.forEach(item => {
                const [ref] = refs[`${item.key}`];
                const myChart = echarts.init(ref);
                const option = generateOption?.[item.key] && generateOption?.[item.key](item.data);
                myChart.hideLoading();
                option && myChart.setOption(option);
            });
        },
        prompt: {
            coherence: data => {
                const coherenceValueList = data.series?.[0]?.data || [];
                let max = 0,
                    maxTopic = "",
                    best = 0,
                    bestTopic = "";
                for (let i = 0; i < coherenceValueList.length; i++) {
                    if (coherenceValueList[i] > max) {
                        max = coherenceValueList[i];
                        maxTopic = data.xAxis[i];
                    }
                }
                for (let i = 0; i < coherenceValueList.length; i++) {
                    if (coherenceValueList[i] > best) {
                        best = coherenceValueList[i];
                        bestTopic = data.xAxis[i];
                    } else {
                        break;
                    }
                }
                return `跑主题一致性时${bestTopic}的平均一致性为${best},${maxTopic}的平均一致性为${max},详细讲述一下为什么选择主题数更偏向选择${bestTopic}`;
            },
            topicWordTree: data => {
                const prompt = data
                    .map(
                        topic =>
                            `${topic.name}中${(topic?.children || [])
                                .map(word => `${word.name}的概率为${word.value}`)
                                .join(",")}`
                    )
                    .join("。\n");
                return `${prompt}。\n根据上述提供的数据来重述各个主题的主题类型并且对各个主题进行相关性分析,同时详细分析这些数据的数据价值并从这些方面对其进行分析。`;
            },
            publishData: data => {
                const publishDataList = data.series?.[0]?.data || [];
                const prompt = publishDataList
                    .slice(0, 50)
                    .map(item => `${item.name}:${item.value}`)
                    .join(";");
                return `接下来我给你发的数据为作者:发文量。${prompt}。\n对作者-发文量进行分析，告诉我你能从这些数据中获得有价值的信息。`;
            },
            authorTopicRelation: data => {
                const links = data?.links || [];
                const nodes = data?.nodes || [];
                const topicMap = {};
                links.forEach(link => {
                    const author = nodes[link?.source];
                    Array.isArray(topicMap[link?.target])
                        ? topicMap[link.target].push({ author: author.name, probability: author.value })
                        : (topicMap[link.target] = [{ author: author.name, probability: author.value }]);
                });
                const prompt = Object.keys(topicMap)
                    .map(key => {
                        const topic = `主题${key}`;
                        const promptItem = (topicMap?.[key] || [])
                            .filter(i => i.probability > 0.6)
                            .sort((a, b) => b.probability - a.probability)
                            .slice(0, 10)
                            .map(item => `${item.author}:${item.probability}`)
                            .join(";");
                        return `在${topic}中作者-主题概率的数据如下：\n${promptItem}。`;
                    })
                    .join("\n");

                return `接下来，我给给你一段数据，这段数据表示的内容为各个作者的文章或者作品的主题概率。\n${prompt}\n告诉我你能从这些数据中获得有价值的信息。`;
            },
        },
    },
};

// Axios 在从客户端在浏览器中发出请求时使用了 XhrAdapter。
// 如果 Axios 在服务器端，将使用 HttpAdapter。
// XhrAdapter 将需要生成XMLHttpRequests但服务器可以生成 HttpRequests。
// 因此有大量关于使用 Axios 在 Node 中处理流的帖子，因为 Node 是一个后端解决方案。
// 在客户端使用 Axios onDownloadProgress
async function askGpt(prompt, obj, key) {
    if (!prompt) return;
    let res = "";
    await axios({
        method: "post",
        data: { prompt },
        responseType: "stream", // 流文件为blob类型
        url: "/model/ai",
        onDownloadProgress({ event }) {
            const textList = (event?.target?.responseText || "")
                .split("data:")
                .filter(i => i)
                .map(item => {
                    try {
                        return `${item}`.toLocaleUpperCase().includes("DONE") ? {} : JSON.parse(item);
                    } catch (error) {
                        return {};
                    }
                });
            const text = textList.map(item => item?.choices?.[0]?.delta?.content || "").join("");
            obj && key && (obj[key] = text);
            res = text;
        },
    });
    return res;
}

function getColor() {
    const colorList = [
        "#0074d9",
        "#ff4136",
        "#ff851b",
        "#444693",
        "#afb4db",
        "#e0861a",
        "#409eff",
        "#e6a23c",
        "#f56c6c",
        "#de773f",
        "#8552a1",
    ];
    return colorList[parseInt(Math.random() * colorList.length)];
}

export { readData, modelMap, askGpt, getColor };
