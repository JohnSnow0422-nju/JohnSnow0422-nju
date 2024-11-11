import fs, { ReadStream } from "fs";
import path from "path";
import XLSX from "xlsx";
import { OrderModel } from "../connect/mongo/model/order";

export async function upload(reader: ReadStream, filePath: string): Promise<boolean | Error> {
    return new Promise((resolve, reject) => {
        const upStream = fs
            .createWriteStream(`${filePath}`)
            .on("close", () => {
                resolve(true);
            })
            .on("error", err => {
                reject(err);
            }); // 创建可写流]
        reader.pipe(upStream);
    });
}

// 1000 ==40kb
const payLevel = [
    { size: 400, unitPrice: 1 },
    { size: 200, unitPrice: 0.38 },
    { size: 100, unitPrice: 0.4 },
    { size: 50, unitPrice: 0.46 },
    { size: 20, unitPrice: 0.5 },
    { size: 5, unitPrice: 0.25 },
];
const modelScale = { lda: 1, dtm: 1.8, bert: 1.6, atm: 1.13 };
export function getFileBill(size: number, model: string) {
    const mb = Number((Number(size) / 100000).toFixed(2));
    const part1Index =
        payLevel.findIndex(item => item.size <= mb) >= 0
            ? payLevel.findIndex(item => item.size <= mb)
            : payLevel.length - 1;
    let part1Price = 0;
    for (let i = part1Index; i < payLevel.length; i += 1) {
        part1Price += payLevel[i].size * payLevel[i].unitPrice;
    }
    const part2Unit = payLevel?.[part1Index - 1]?.unitPrice || payLevel?.[part1Index]?.unitPrice || 0.3;
    const part2Size = Math.max(mb - payLevel?.[part1Index]?.size || 0, 0);
    const part2Price = part2Unit * part2Size;
    const resultPrice = Math.max(part2Price + part1Price, 8);
    return Number(Number(`${resultPrice * (modelScale?.[model] || 1)}`).toFixed(1)) * 10;
}

export function emptyDir(dirname: string) {
    try {
        const fileList = fs.readdirSync(dirname) || [];
        fileList.forEach(filename => {
            const currntFile = `${dirname}/${filename}`;
            if (fs.statSync(currntFile).isDirectory()) {
                //同步读取文件夹文件，如果是文件夹，则函数回调
                emptyDir(currntFile);
            } else {
                fs.unlinkSync(currntFile); //是指定文件，则删除
            }
        });
        fs.rmdirSync(dirname); //清除文件夹
    } catch (error) {
        console.log("回退出错");
        console.log(error);
    }
}

export function readExcel(dirname: string): any {
    const workBook = XLSX.readFile(dirname, { type: "binary" });
    return Object.keys(workBook.Sheets).map(sheetName => ({
        sheetName,
        data: XLSX.utils.sheet_to_json(workBook.Sheets[sheetName]),
    }));
}

export function parser(data: OrderModel) {
    const fileDir = path.resolve(import.meta.dirname, "../../../output");
    const modelMap = {
        dtm: (params: OrderModel) => {
            const dir = `${fileDir}/${params?._id}`;

            //处理词频
            const [{ data: splitWordList }] = readExcel(`${dir}/filter.xlsx`);
            const splitWordMap = {};
            const timeSlice = [];
            splitWordList.forEach(item => {
                const list = (item?.word_split || "").split(",");
                const time = item.time;
                if (splitWordMap?.[time]) {
                    splitWordMap[time].push(...list);
                } else {
                    splitWordMap[time] = [...list];
                }
                const target = timeSlice.find(i => i.time === time);
                if (target) {
                    target.num++;
                } else {
                    timeSlice.push({ time, num: 1 });
                }
            });
            const wordFrequency = Object.keys(splitWordMap).map(key => {
                const list = [];
                splitWordMap[key].forEach(word => {
                    const target = list.find(item => item?.name === word);
                    if (target) {
                        target.value += 1;
                    } else {
                        list.push({ name: word, value: 1 });
                    }
                });
                return { name: key, children: list.sort((a, b) => (b?.value || 0) - (a?.value || 0)).slice(0, 10) };
            });
            //处理一致性
            const [{ data: coherenceData }] = readExcel(`${dir}/coherence.xlsx`);
            const coherenceMap = {};
            const coherencexAxis = [];
            let coherenceyAxisMin = 1;
            coherenceData.forEach(item => {
                if (Array.isArray(coherenceMap?.[item?.["topic_num"]])) {
                    coherenceMap[item["topic_num"]].push({
                        time: Number(item["time"]),
                        cv: Number(item["coherence_value"]),
                    });
                } else {
                    coherenceMap[item["topic_num"]] = [
                        { time: Number(item["time"]), cv: Number(item["coherence_value"]) },
                    ];
                }
                coherenceyAxisMin = Math.min(Number(item["coherence_value"]), Number(coherenceyAxisMin));
                coherencexAxis.push(item["time"]);
            });
            const coherenceSeries = Object.keys(coherenceMap).map(key => ({
                name: `主题数: ${key}`,
                data: coherenceMap[key].sort((a, b) => a.time - b.time).map(item => Number(item.cv.toFixed(3))),
            }));
            timeSlice.sort((a, b) => a.time - b.time);
            const timeSort = timeSlice.map(i => i.time);
            const timeNum = timeSlice.map(i => i.num);
            //处理主题强度变化
            const [{ data: docTopic }] = readExcel(`${dir}/不同文档的主题分布.xlsx`);
            let start = 0;
            let themeAxisMin = 1;
            let themeList = [];
            const themeMiddleList = timeNum
                .map(item => {
                    const res = docTopic.slice(start, item + start);
                    start += item;
                    return res;
                })
                .map(item => {
                    const middleMap = {};
                    item.forEach(doc => {
                        Object.keys(doc).forEach(key => {
                            if (Number(key) < 0) return;
                            if (middleMap?.[key]) {
                                middleMap[key] += Number(doc[key]);
                            } else {
                                middleMap[key] = Number(doc[key]);
                            }
                        });
                    });
                    Object.keys(middleMap).forEach(key => {
                        middleMap[key] = Number((middleMap[key] / item.length).toFixed(4));
                        themeAxisMin = Math.min(themeAxisMin, middleMap[key]);
                    });
                    themeList = Object.keys(middleMap).map(key => `${key}`);
                    return middleMap;
                });
            const themeSeries = themeList.map(key => {
                const themeSeqList = [];
                themeMiddleList.forEach(item => {
                    themeSeqList.push(item[key]);
                });
                return { name: `主题${key}`, data: themeSeqList };
            });
            //处理情感分析
            const [{ data: sentiment }] = readExcel(`${dir}/情感分析表.xlsx`);
            const sentimentMiddleList = timeSort
                .map(time => sentiment.filter(item => `${item.time}` === `${time}`))
                .map((itemInTime, index) => {
                    const map = {};
                    const time = timeSort[index];
                    itemInTime.forEach(item => {
                        if (map?.[item["max_idx"]]) {
                            map[item["max_idx"]].value += item.sentiment;
                            map[item["max_idx"]].num += 1;
                        } else {
                            map[item["max_idx"]] = { value: item.sentiment, num: 1 };
                        }
                    });
                    Object.keys(map).forEach(key => {
                        map[key] = map[key].value / map[key].num;
                    });
                    return { time, sentiment: map };
                });
            const sentimentData = sentimentMiddleList
                .map((time, index) => {
                    return Object.keys(time.sentiment).map(key => {
                        return [index, Number(key), Number(Number(time.sentiment[key]).toFixed(2))];
                    });
                })
                .flat();
            //处理不同时间段主题词的分布概率
            const topicWordByTime = readExcel(`${dir}/不同时期主题分布.xlsx`);
            const topicWordTree = topicWordByTime.map(item => ({
                children: (item.data || []).map((topicWord, topicIndex) => ({
                    name: `主题${topicIndex}`,
                    children: Object.keys(topicWord)
                        .map(key => {
                            const filterString = topicWord[key].slice(1, (topicWord[key]?.length || 2) - 1);
                            const [word, value] = filterString.split(",");
                            const first = `${word}`.indexOf("'");
                            const last = `${word}`.lastIndexOf("'");
                            return {
                                name: `${word}`.slice(first + 1, last),
                                value: Number(Number(value).toFixed(5)),
                            };
                        })
                        .slice(0, 10),
                })),
                name: item.sheetName,
            }));
            return [
                {
                    title: "一致性",
                    key: "coherence",
                    data: {
                        series: coherenceSeries,
                        xAxis: Array.from(new Set(coherencexAxis)).sort((a, b) => a - b),
                        yMin: Number((coherenceyAxisMin * 0.98).toFixed(2)),
                    },
                },
                {
                    title: "主题强度变化趋势",
                    key: "theme",
                    data: {
                        yMin: Number((themeAxisMin * 0.98).toFixed(2)),
                        series: themeSeries,
                        xAxis: timeSort,
                    },
                },
                { title: "词频统计树", height: 680, key: "wordFrequency", data: wordFrequency },
                { title: "主题词分布概率", height: 600, key: "topicWordTree", data: topicWordTree },
                {
                    title: "情感分析",
                    key: "sentiment",
                    height: 400,
                    data: {
                        data: sentimentData,
                        xAxis: timeSort,
                        yAxis: themeList.map(key => `主题${key}`),
                    },
                },
            ];
        },
        lda: (params: OrderModel) => {
            const dir = `${fileDir}/${params?._id}`;
            //处理一致性
            const [{ data: coherenceData }] = readExcel(`${dir}/coherence.xlsx`);
            const coherenceSeries = coherenceData.map(item => Number(item["coherence_value"]).toFixed(4));
            //处理主题强度
            const [{ data: docTopic }] = readExcel(`${dir}/不同文档的主题分布.xlsx`);
            const middleMap = {};
            docTopic.forEach(item => {
                Object.keys(item).forEach(key => {
                    if (middleMap[key]) {
                        middleMap[key] += Number(item[key]);
                    } else {
                        middleMap[key] = Number(item[key]);
                    }
                });
            });
            let themeMax = 0;
            Object.keys(middleMap).forEach(key => {
                middleMap[key] = (middleMap[key] / docTopic.length).toFixed(4);
                themeMax = Math.max(middleMap[key], themeMax);
            });
            const indicator = [];
            const themeSeries = [];
            Object.keys(middleMap).forEach(key => {
                indicator.push({ name: `主题${key}`, max: Number((themeMax * 1.2).toFixed(2)) });
                themeSeries.push(Number(middleMap[key]));
            });
            //处理词频
            const [{ data: splitWordList }] = readExcel(`${dir}/filter.xlsx`);
            const splitwordFrequency = [];
            splitWordList
                .map(item => item["word_split"].split(","))
                .flat()
                .forEach(item => {
                    const target = splitwordFrequency.find(i => i.name === item);
                    if (target) {
                        target.value += 1;
                    } else {
                        splitwordFrequency.push({ value: 1, name: item });
                    }
                });
            //处理主题词分布概率
            const [{ data: topicWord }] = readExcel(`${dir}/主题词分布概率.xlsx`);
            const topicWordTree = topicWord.map((topicWordItem, topicIndex) => ({
                name: `主题${topicIndex}`,
                children: Object.keys(topicWordItem)
                    .map(key => {
                        const [value, word] = topicWordItem[key].split("*");
                        const first = `${word}`.indexOf("'");
                        const last = `${word}`.lastIndexOf("'");
                        return {
                            value: Number(Number(value).toFixed(5)),
                            name: `${word}`.slice(first + 1, last),
                        };
                    })
                    .slice(0, 10),
            }));
            //处理情感分析
            const [{ data: sentiment }] = readExcel(`${dir}/情感分析表.xlsx`);
            const data = [],
                themeNum = [];
            sentiment.forEach(item => {
                themeNum.push(Number(item["max_idx"]));
                const sentiment = Number(Number(item["sentiment"]).toFixed(2));
                const target = data.find(
                    i => Number(i?.[0]) === Number(item["max_idx"]) && Number(i?.[1]) === sentiment
                );
                if (target) {
                    target[2] += 1;
                } else {
                    data.push([Number(item["max_idx"]), sentiment, 1]);
                }
            });
            return [
                {
                    key: "coherence",
                    title: "一致性",
                    data: {
                        xAxis: coherenceData.map(item => `主题数: ${item["topic_num"]}`),
                        series: [{ data: coherenceSeries, type: "line", smooth: true }],
                    },
                },
                {
                    key: "theme",
                    title: "主题强度雷达图",
                    data: { indicator: indicator, series: [{ name: "不同主题的主题强度", value: themeSeries }] },
                },
                {
                    key: "wordFrequency",
                    title: "词频统计",
                    height: 600,
                    data: { series: splitwordFrequency.sort((a, b) => b.value - a.value).splice(0, 50) },
                },
                { height: 600, key: "topicWordTree", title: "主题词分布概率", data: topicWordTree },
                {
                    key: "sentiment",
                    height: Array.from(new Set(themeNum)).length * 100,
                    title: "文本情感分布",
                    data: { themeNum: Array.from(new Set(themeNum)).sort((a, b) => a - b), data },
                },
            ];
        },
        bert: (params: OrderModel) => {
            const dir = `${fileDir}/${params?._id}`;
            const [{ data: docTopic }] = readExcel(`${dir}/主题词分布概率.xlsx`);
            const themeTopic = docTopic.map(item => `主题${item.topic}`);
            const themeCount = docTopic.map(item => Number(item.count));
            const topicWordTree = docTopic.map(item => {
                const name = `主题${item.topic}`;
                const children = (`${item["word_value"]}`.split(",") || []).map(word => {
                    const [name, value] = `${word}`.split(":");
                    return { name, value };
                });
                return { name, children };
            });
            const [{ data: sentiment }] = readExcel(`${dir}/情感分析表.xlsx`);
            const data = [],
                themeNum = [];
            sentiment.forEach(item => {
                themeNum.push(Number(item["max_idx"]));
                const senti = Number(Number(item["sentiment"]).toFixed(2));
                const target = data.find(i => Number(i?.[0]) === Number(item["max_idx"]) && Number(i?.[1]) === senti);
                if (target) {
                    target[2] += 1;
                } else {
                    data.push([Number(item["max_idx"]), senti, 1]);
                }
            });
            const [{ data: splitWordList }] = readExcel(`${dir}/filter.xlsx`);
            const splitwordFrequency = [];
            splitWordList
                .map(item => item["word_split"].split(","))
                .flat()
                .forEach(item => {
                    const target = splitwordFrequency.find(i => i.name === item);
                    if (target) {
                        target.value += 1;
                    } else {
                        splitwordFrequency.push({ value: 1, name: item });
                    }
                });
            return [
                {
                    key: "theme",
                    title: "不同主题文档数量",
                    data: { xAxis: themeTopic, series: { data: themeCount } },
                },
                { height: 600, key: "topicWordTree", title: "主题词分布概率", data: topicWordTree },
                {
                    key: "wordFrequency",
                    title: "词频统计",
                    height: 600,
                    data: { series: splitwordFrequency.sort((a, b) => b.value - a.value).splice(0, 50) },
                },
                {
                    key: "sentiment",
                    height: Array.from(new Set(themeNum)).length * 100,
                    title: "文本情感分布",
                    data: { themeNum: Array.from(new Set(themeNum)).sort((a, b) => a - b), data },
                },
            ];
        },
        atm: (params: OrderModel) => {
            const dir = `${fileDir}/${params?._id}`;
            //处理一致性
            const [{ data: coherenceData }] = readExcel(`${dir}/coherence.xlsx`);
            const coherenceSeries = coherenceData.map(item => Number(item["coherence_value"]).toFixed(4));
            //处理作者发文量
            const [{ data: filterData }] = readExcel(`${dir}/filter.xlsx`);
            const publishMap = {};
            filterData.forEach((item: { author: string }) => {
                item.author
                    .split(";")
                    .map(i => i.trim())
                    .forEach(author => {
                        publishMap[author] = (publishMap[author] || 0) + 1;
                    });
            });
            const publishMiddleData = Object.keys(publishMap)
                .map(name => ({ name, value: publishMap[name] }))
                .sort((a, b) => b.value - a.value);
            const publishSliceNum = Math.max(Math.round((Object.keys(publishMap).length * 4) / 5), 50);
            const slice1 = publishMiddleData.slice(0, publishSliceNum);
            const other = { name: "其他", value: 0 };
            publishMiddleData.slice(publishSliceNum).forEach(item => {
                other.value += item.value;
            });
            const publishData = [...slice1, other];
            //处理主题词分布概率
            const [{ data: topicWord }] = readExcel(`${dir}/主题词分布概率.xlsx`);
            const topicWordTree = topicWord.map((topicWordItem, topicIndex) => ({
                name: `主题${topicIndex}`,
                children: Object.keys(topicWordItem)
                    .map(key => {
                        const [value, word] = topicWordItem[key].split("*");
                        const first = `${word}`.indexOf("'");
                        const last = `${word}`.lastIndexOf("'");
                        return {
                            value: Number(Number(value).toFixed(5)),
                            name: `${word}`.slice(first + 1, last),
                        };
                    })
                    .slice(0, 10),
            }));
            //处理主题-作者聚类
            const [{ data: authorTopic }] = readExcel(`${dir}/不同作者的主题分布.xlsx`);
            const categories = topicWord.map((_item, index) => ({
                name: `主题${index}`,
                id: index,
                symbolSize: 30,
                category: index,
            }));
            const links = [];
            const nodes = [...categories];
            authorTopic.forEach((item: { [x: string]: string }) => {
                const author = item[0];
                item[1].split(";").forEach(i => {
                    const [topicIndex, topicValue] = i.split(":");
                    const id = nodes.length;
                    nodes.push({
                        name: `${author}`,
                        id,
                        category: Number(topicIndex),
                        value: Number(topicValue),
                        symbolSize: Math.max(Number(topicValue), 0.3) * 20,
                    });
                    links.push({ source: id, target: Number(topicIndex) });
                });
            });
            return [
                {
                    key: "coherence",
                    title: "一致性",
                    data: {
                        xAxis: coherenceData.map(item => `主题数: ${item["topic_num"]}`),
                        series: [{ data: coherenceSeries, type: "line", smooth: true }],
                    },
                },
                {
                    key: "publishData",
                    title: "作者发文数量",
                    data: {
                        series: [{ data: publishData, label: { show: Math.random() < 0.5 }, type: "pie" }],
                    },
                },
                { key: "topicWordTree", title: "主题词分布概率", data: topicWordTree, height: 600 },
                {
                    key: "authorTopicRelation",
                    title: "作者主题聚类",
                    data: {
                        categories,
                        links,
                        nodes,
                    },
                    height: 600,
                },
            ];
        },
    };
    try {
        return modelMap[data?.params?.model](data);
    } catch (error) {
        console.log(error);
        throw new Error("模型未跑完.zzz");
    }
}

export const AI_KEY = "sk-sjRUr60Wr5itNQpkfoOIRRP2zhZF3CyEWCkeRe3HlmCDFIMW";
