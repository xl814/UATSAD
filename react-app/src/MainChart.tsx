import * as d3 from "d3";
import { useRef, useEffect, useState } from "react";
import {Card, Col, Row, Button} from 'antd';

import {CheckOutlined, CloseOutlined} from '@ant-design/icons'; 
import http from "./https";

import "./MainChart.css";

import Epistemic_anim from "./Multiples";
import { upperQuartile } from "./utils";

export interface Item {
    x: number,
    y: number
}

interface LineChartProps {
    // data param
    data: Array<Item>,
    smooth_level?: number,

    // plot param
    width?: number,
    height?: number,
    marginTop?: number,
    marginRight?: number,
    marginBottom?: number,
    marginLeft?: number,
}

interface MainChartProps {
    data: Array<Item>,
    dataset: string,
    // showEpis: boolean,
    // showDownsampled: boolean,
    // showAleatoric: boolean,
    // showAnomaly: boolean,

    // epis_50: [number, number][];
    // epis_95: [number, number][];
    // downsampled_data: Array<Item>;
    // aleatoric: [number, number][];
    // anomaly_response: {};


    // plot param
    width?: number,
    height?: number,
    marginTop?: number,
    marginRight?: number,
    marginBottom?: number,
    marginLeft?: number,
}

interface EpisChartProps{
    epis_upper_95: Array<Item>,
    epis_lower_95: Array<Item>,
    epis_upper_50: Array<Item>,
    epis_lower_50: Array<Item>,
}

interface AleaChartProps{
    alea: [number, number][],
}

interface EpisItem{
    upper: number,
    lower: number
}

let epis_50: [number, number][] = [];
let epis_95: [number, number][] = [];
let downsampled_data: Array<Item> = []
let selected_range1:[number, number] = [0, 0]
let selected_range2:[number, number] = [0, 0]
let selected_range3:[number, number] = [0, 0]
let current_brush_index = 0;
const epis_view_height = 50;
const alea_view_height = 100;

export const colorScale = d3.scaleSequential(d3.interpolateRgb("#fff042", "#ff4254")).domain([0, 1]);

function Legend({width=900, height=15}){
    useEffect(()=>{        

        // 创建渐变的颜色条
        const color_legend = d3.select(".anomaly_legend");

        const uncertainty_legend = d3.select(".Uncertainty_legend");
    
        const gradient = color_legend.append("defs")
            .append("linearGradient")
            .attr("id", "gradient")
            .attr("x1", "0%")
            .attr("x2", "100%");
    
        gradient.append("stop")
            .attr("offset", "0%")
            .attr("stop-color", colorScale(0));
    
        gradient.append("stop")
            .attr("offset", "100%")
            .attr("stop-color", colorScale(1));
    
        // 绘制矩形并应用渐变
        color_legend.append("rect")
            .attr("x", 50) // 50 is distance from text label `anomaly score`
            .attr("y", 5)
            .attr("width", 150)
            .attr("height", height)
            .style("fill", "url(#gradient)");
        
        // epistemic 
        d3.select(".uncertainty_legend").append("rect")
            .attr("x", 65)
            .attr("y", 5)
            .attr("width",40)
            .attr("height", height)
            .style("fill", "steelblue")
            .style("opacity", 0.3);

        d3.select(".uncertainty_legend").append("text")
                .attr("x", 80)
                .attr("y", height / 2 + 6)
                .attr("text-anchor", "middle")
                .attr("font-size", "10")
                .text("95%")
                .attr("fill", "white")

        d3.select(".uncertainty_legend").append("rect")
            .attr("x", 105) // 65 + 40
            .attr("y", 5)
            .attr("width",40)
            .attr("height", height)
            .style("fill", "steelblue")
            .style("opacity", 0.7);

        d3.select(".uncertainty_legend").append("text")
            .attr("x", 120) // 105 + 15
            .attr("y", height / 2 + 6)
            .attr("text-anchor", "middle")
            .attr("font-size", "10")
            .text("50%")
            .attr("fill", "white")

        // Aleatoric
        d3.select(".aleatoric_legend").append("rect")
        .attr("x", 60)
        .attr("y", 5)
        .attr("width",40)
        .attr("height", height)
        .style("fill", "lightgreen")
        .attr("opacity", 0.3)

    }, [])
   


    return (
        <svg width={width} height={height}>
            
            {/* <text x={250} y={height / 2 + 5}  textAnchor="middle" fontSize={12}> Epistemic Uncertainty</text>
            <text x={10} y={height / 2 + 5}  textAnchor="middle" fontSize={12}> Aleatoric Uncertainty</text> */}
            <g className="anomaly_legend" transform={`translate(${100}, ${0})`}>
                <text  y={height / 2 + 5} textAnchor="middle" fontSize={12}> Anomaly Score</text>
            </g>
            <g className="uncertainty_legend" transform={`translate(${400}, ${0})`}> 
                <text  y={height / 2 + 5} textAnchor="middle" fontSize={12}> Epistemic Uncertainty</text>
            </g>
            <g className="aleatoric_legend" transform={`translate(${650}, ${0})`}> 
                <text  y={height / 2 + 5} textAnchor="middle" fontSize={12}> Aleatoric Uncertainty</text>
            </g>
        </svg>
    )
}



function Aleatoric({alea, width = 900, height, marginTop = 3, marginRight = 20, marginBottom = 5, marginLeft = 40}: any){
    console.log(alea)

    const svg_ref = useRef(null);
    let maxAlea = d3.max(alea, (d: Item)=>d.y) as number
    // const gx = useRef<null|SVGGElement>(null);
    // const gy = useRef<null|SVGGElement>(null);
    const xAxis = (g: any, x: any) => g.call(d3.axisBottom(x).ticks(10) as any);
    const yAxis = (g: any, y: any) => g.call(d3.axisLeft(y).ticks(3) as any);
    const x = d3.scaleLinear(d3.extent(alea, (d: Item)  => d.x) as [number, number], [marginLeft, width - marginRight]);
    // const y = d3.scaleLinear(d3.extent(alea, (d: Item) => d.y) as [number, number], [height - marginBottom, marginTop]);
    const y = d3.scaleLinear().domain([
        // d3.min(mult_epis, (d: Array<Item>) => d3.min(d, (d: Item) => d.y)) as number,
        0,
        maxAlea
    ]).range([height - marginBottom, marginTop])
    // const area = d3.area()
    //     .x((d, i) => x(i))
    //     .y0((d) => y(d[0]))
    //     .y1((d) => y(d[1]));

    const area = (data: Array<[number, number]>, y:any) => d3.area()
        .x((d, i) => x(i))
        .y0((d) => y(d[0]))
        .y1((d) => y(d[1]))(data);

    let area_alea: [number, number][] = [] // convert alea to area data 
    let alea_y: number[] = []; 
    alea.forEach((item: any) => {
        area_alea.push([0, item.y]);
        alea_y.push(item.y);
    });

    let upper_quartile = upperQuartile(alea_y);
    useEffect(() => {
        d3.select(".gx").selectChild("path").attr("stroke", "gray");
        d3.select(".gy").selectChild("path").attr("stroke", "gray")
        const refline_alea = d3.select(".ref_line4alea")
            .attr("y1", 0)
            .attr("y2", height - marginBottom)
            .attr("stroke", "gray")
            .attr("stroke-width", 1)
            .attr("stroke-dasharray", "4 2")
            .attr("visibility", "hidden");

        const quartile_line = d3.select(".upper_quartile_alea")
            .attr("y1", y(upper_quartile)) // y(0.75 * maxAlea))
            .attr("y2", y(upper_quartile))
            .attr("x1", marginLeft)
            .attr("x2", width - marginRight)
            .attr("stroke", "gray")
            .attr("stroke-width", 1)
            .attr("stroke-dasharray", "4 2")

        const gx = d3.select(".gx").call(xAxis, x);
        const gy = d3.select(".gy").call(yAxis, y);



        const path = d3.select(".alea_content");
        
        const zoomed = (event: any) => {
            // const xz = event.transform.rescaleX(x);
            const yz = event.transform.rescaleY(y);
            let alea_y: number[] = []; 
            alea.forEach((item: any) => {
                if(item.y >= yz.domain()[0] && item.y <= yz.domain()[1])
                    alea_y.push(item.y);
            });
            upper_quartile = upperQuartile(alea_y);
          
            path.attr("d", area(area_alea, yz));
            quartile_line
                .attr("y1", yz(upper_quartile))
                .attr("y2", yz(upper_quartile))
            // gx.call(xAxis, xz);
            gy.call(yAxis, yz)
        };
        const zoom = d3.zoom()
            .scaleExtent([1, 32])
            .extent([[marginLeft, marginTop], [width - marginRight, height - marginBottom]])
            .translateExtent([[marginLeft, marginTop], [width - marginRight, height - marginBottom]])
            .on("zoom", zoomed);
            d3.select(svg_ref.current).call(zoom as any);

    }, [])


    return (
        <svg ref={svg_ref} width={width} height={height}>
            <g className="gx" transform={`translate(0, ${height - marginBottom})`} />
            <g className="gy" transform={`translate(${marginLeft}, 0)`} />
            <g >

                <clipPath id="clip_alea">
                    <rect x={marginLeft} y={marginTop} width={width - marginLeft - marginRight} height={height - marginTop - marginBottom}></rect>
                </clipPath>
                <path clipPath="url(#clip_alea)" className="alea_content" fill="lightgreen" opacity="0.3" d={area(area_alea, y) as string} />
                {/* <rect x={0} y={0} width={width}  height={height} fill="#f0f0f085" /> */}
                {/* <path className="alea-content" fill="lightgreen" opacity="1" d={area(area_alea) as string} /> */}
                <line className="ref_line4alea" />
                <line className="upper_quartile_alea" />
                <text x={width - marginRight - 40} y={marginTop + 40} textAnchor="middle" fontSize="10" fill="gray" > 75th percentile </text>
                <text x={marginLeft + 20} y={marginTop + 4} textAnchor="middle" fontSize="10" fill="black" > Aleatoric </text>
            </g>
        </svg>
    )
}

function Epistemic({epis, width = 900, height, marginTop = 3, marginRight = 20, marginBottom = 5, marginLeft = 40}: any){
    let maxEpis = d3.max(epis, (d: Item)=>d.y) as number ; 
    const gx = useRef<null|SVGGElement>(null);
    const gy = useRef<null|SVGGElement>(null);
    const svg = useRef(null);

    const x = d3.scaleLinear(d3.extent(epis, (d: Item)  => d.x) as [number, number], [marginLeft, width - marginRight]);
    // const y = d3.scaleLinear(d3.extent(epis, (d: Item) => d.y) as [number, number], [marginTop, height - marginBottom]);
    const y = d3.scaleLinear().domain([0, maxEpis]).range([height - marginBottom, marginTop])  // d3.min(mult_epis, (d: Array<Item>) => d3.min(d, (d: Item) => d.y)) as number,
    const xAxis = (g: any, x: any) => g.call(d3.axisBottom(x).ticks(10) as any);
    const yAxis = (g: any, y: any) => g.call(d3.axisLeft(y).ticks(3) as any);
    // useEffect(() => void d3.select(gx.current).call(d3.axisBottom(x).ticks(10) as any), [gx, x]);
    // useEffect(() => void d3.select(gy.current).call(d3.axisLeft(y).ticks(3) as any ), [gy, y]);

    const area = (data: Array<[number, number]>, y:any) => d3.area()
        .x((d, i) => x(i))
        .y0((d) => y(d[0]))
        .y1((d) => y(d[1]))(data);



    let area_epis: [number, number][] = [] // convert alea to area data 
    let epis_y: number[] = []
    epis.forEach((item: any) => {
        area_epis.push([0, item.y]);
        epis_y.push(item.y);
    });

    let upper_quartile = upperQuartile(epis_y);


    useEffect(() => {
        // d3.select(ref_svg.current).append("g")
        //     .attr("class", "brush")
        //     .call(brush);
        d3.select(gx.current).call(xAxis, x);
        d3.select(gy.current).call(yAxis, y);
        d3.select(gx.current).selectChild("path").attr("stroke", "gray");
        d3.select(gy.current).selectChild("path").attr("stroke", "gray")

        const refline_epis = d3.select(".ref_line4epis")
            .attr("y1", 0)
            .attr("y2", height - marginBottom)
            .attr("stroke", "gray")
            .attr("stroke-width", 2)
            .attr("stroke-dasharray", "4 2")
            .attr("visibility", "hidden");

        const quartile_line = d3.select(".upper_quartile_epis")
            .attr("y1", y(upper_quartile))
            .attr("y2", y(upper_quartile))
            .attr("x1", marginLeft)
            .attr("x2", width - marginRight)
            .attr("stroke", "gray")
            .attr("stroke-width", 1)
            .attr("stroke-dasharray", "4 2")



        const path = d3.select(".epis_content");
        
        const zoomed = (event: any) => {
            // const xz = event.transform.rescaleX(x);
            const yz = event.transform.rescaleY(y);
            let epis_y: number[] = []
            epis.forEach((item: any) => {
                if (item.y >= yz.domain()[0] && item.y <= yz.domain()[1])
                    epis_y.push(item.y);
            });
        
            upper_quartile = upperQuartile(epis_y);
            path.attr("d", area(area_epis, yz));
            quartile_line
                .attr("y1", yz(upper_quartile))
                .attr("y2", yz(upper_quartile))
            // gx.call(xAxis, xz);
            d3.select(gy.current).call(yAxis, yz)
        };
        const zoom = d3.zoom()
            .scaleExtent([1, 32])
            .extent([[marginLeft, marginTop], [width - marginRight, height - marginBottom]])
            .translateExtent([[marginLeft, marginTop], [width - marginRight, height - marginBottom]])
            .on("zoom", zoomed);
        d3.select(svg.current).call(zoom as any);
    }, [])


    return (
        <svg className="epis_svg" ref={svg} width={width} height={height}>
            <g ref={gx} className="gx" transform={`translate(0, ${height - marginBottom})`} />
            <g ref={gy}  transform={`translate(${marginLeft}, 0)`} />
            <g>
                <clipPath id="clip_epis">
                    <rect x={marginLeft} y={marginTop} width={width - marginLeft - marginRight} height={height - marginTop - marginBottom}></rect>
                </clipPath>
                {/* <rect x={0} y={0} width={width}  height={height} fill="#f0f0f085" /> */}
                <path clipPath="url(#clip_epis)" className="epis_content" fill="steelblue" opacity="0.3" d={area(area_epis, y) as string} />
                <line className="ref_line4epis" />
                <line className="upper_quartile_epis" />
                <text x={width - marginRight - 40} y={marginTop + 8} textAnchor="middle" fontSize="10" fill="gray" > 75th percentile </text>
                <text x={marginLeft + 22} y={marginTop + 4} textAnchor="middle" fontSize="10" > Epistemic </text>
            </g>
        </svg>
    )
}





export default function MainChart({data, dataset, width = 900, height = 200, marginTop = 10, marginRight = 20, marginBottom = 20, marginLeft = 40}: MainChartProps) {

    if (data === undefined) {
        return <div>data is empty!</div>
    }

    // const [showEpis, setShowEpis] = useState(false);
    const [alea, setAlea] = useState<[number, number][]>([]);
    const [epis, setEpis] = useState<[number, number][]>([]);
    const [showSampledData, setShowSampledData] = useState(false);
    const [multiplesIndex, setMultiplesIndex] = useState(0);
    const [multi_epis1, setMultiEpis1] = useState(null);
    const [multi_epis2, setMultiEpis2] = useState(null);
    const [multi_epis3, setMultiEpis3] = useState(null);


    const ref_svg = useRef<null|SVGAElement>(null);
    const gx = useRef<null|SVGGElement>(null);
    const gy = useRef<null|SVGGElement>(null);
    // const x = d3.scaleUtc(d3.extent(data, (d: Item) => new Date(d.x)), [marginLeft, width - marginRight]);
    let y_domain = d3.extent(data, (d: Item) => d.y);
    y_domain[1] = y_domain[1] as number + 0.5; // expand y domain for placing more space
    y_domain[0] = y_domain[0] as number - 1;

    const x = d3.scaleLinear(d3.extent(data, (d: Item)  => d.x) as [number, number], [marginLeft, width - marginRight]);
    const y = d3.scaleLinear(y_domain as [number, number], [height - marginBottom, marginTop]);

    // const line = d3.line((d: Item) => x(new Date(d.x)), (d: Item) => y(d.y));

    const area = d3.area()
        .x((d, i) => x(i))
        .y0((d) => y(d[0]))
        .y1((d) => y(d[1]));
    
    let current_brush_loc = {
        x0: 0,
        x1: 0,
        y0: 0,
        y1: 0
    };
    let tooltip_loc: [number, number] = [0, 0] // [left, top]
    const brush = d3.brush()
        .on("start", (event)=>{
            let brush_tooltip = null;
            if(current_brush_index === 0)
                brush_tooltip = d3.select(".brush_tooltip_1");
            else if(current_brush_index === 1)
                brush_tooltip = d3.select(".brush_tooltip_2");
            else if(current_brush_index === 2)
                brush_tooltip = d3.select(".brush_tooltip_3");
            // console.log(event.pageX, event.pageY)
            if (current_brush_index < 3)
                (brush_tooltip as any).style("visibility", "hidden")
        })
        .on(" end", (event)=>{
           
            console.log(event)
            if(event.selection && showSampledData && current_brush_index < 3){
                const [[x0, y0], [x1, y1]] = event.selection;
                current_brush_loc.x0 = x0;
                current_brush_loc.y0 = y0;
                current_brush_loc.x1 = x1;
                current_brush_loc.y1 = y1;
                let brush_tooltip = null;
                let y0_copy = y0 + 10; 
                if(epis.length !== 0)
                    y0_copy += epis_view_height; // 40 is epis/alea coponent's height 
                if(alea.length !== 0)
                    y0_copy += alea_view_height;
                if(current_brush_index === 0)
                    brush_tooltip = d3.select(".brush_tooltip_1");
                else if(current_brush_index === 1)
                    brush_tooltip = d3.select(".brush_tooltip_2");
                else if(current_brush_index === 2)
                    brush_tooltip = d3.select(".brush_tooltip_3");
                // console.log(event.pageX, event.pageY)
                if (current_brush_index < 3)
                    (brush_tooltip as any)
                        .style("visibility", "visible")
                        .style("left", x1 + "px") // 112 is margin-left of div[#root]
                        .style("top", ((brush_tooltip as any).node() as HTMLElement).getBoundingClientRect().height + y0_copy + "px");
                    tooltip_loc = [x1, ((brush_tooltip as any).node() as HTMLElement).getBoundingClientRect().height + y0_copy];
            }
        })
    if (ref_svg.current !== null)    
        d3.select(ref_svg.current).call(brush as any);
    
    const line = d3.line((d: Item) => x(d.x), (d: Item) => y(d.y));
    // data.forEach(d => {console.log(x(d.x))});
    if (line === null) {
        return <div> line is null! </div>
    }



    function handleEpisMousemove(pos_x: number){
        const refline_epis = d3.select(".ref_line4epis")
            .attr("x1", pos_x)
            .attr("x2", pos_x)
            .attr("visibility", "visible");
    }


    function handleAleaMousemove(pos_x: number){
        const refline_epis = d3.select(".ref_line4alea")
            .attr("x1", pos_x)
            .attr("x2", pos_x)
            .attr("visibility", "visible");
    }
    // useEffect(() => void d3.select(gx.current).call(d3.axisBottom(x).ticks(width / 100).tickFormat(d3.utcFormat("%B %d, %Y"))), [gx, x]);
    useEffect(() => void d3.select(gx.current).call(d3.axisBottom(x) as any), [gx, x]);
    useEffect(() => void d3.select(gy.current).call(d3.axisLeft(y) as any ), [gy, y]);
    useEffect(() => {
        d3.select(gx.current).selectChild("path").attr("stroke", "gray");
        d3.select(gy.current).selectChild("path").attr("stroke", "gray")

        const refline = d3.select(".ref_line").append("line")
            .attr("y1", marginTop)
            .attr("y2", height - marginBottom)
            .attr("stroke", "black")
            .attr("stroke-width", 2)
            .attr("stroke-dasharray", "4 2")
            .attr("visibility", "hidden");

        d3.select(ref_svg.current).on("mousemove", (e) => {
            const [xPos, yPos] = d3.pointer(e);
            refline.attr("x1", xPos).attr("x2", xPos).attr("visibility", "visible");
            handleEpisMousemove(xPos);
            handleAleaMousemove(xPos);
        }).on("mouseleave", () => {
            refline.attr("visibility", "hidden");
        });

        // remove svg elements
        return () => {
            d3.select(ref_svg.current).selectAll("*").remove();
        }

    }, [])

    function handleEpisClick(e: any) {
        e.preventDefault();
        http.get('/api/epis')
        .then((response) => {
            // console.log(response.data);
            const ret = response.data;
            for (let i = 0; i < ret.epis_upper_50.length; i++) {
                epis_50.push([ret.epis_lower_50[i].y, ret.epis_upper_50[i].y]);
            }
            for (let i = 0; i < ret.epis_upper_95.length; i++) {
                epis_95.push([ret.epis_lower_95[i].y, ret.epis_upper_95[i].y]);
            }

            console.log(response.data);
            d3.select(".epis_50").attr("d", area(epis_50));
            d3.select(".epis_95").attr("d", area(epis_95));

            setEpis(ret.epis)

        }).catch((error) => {
            console.log(error);
        });
        
    }


    function handleDownsampleClick(e: any){
        e.preventDefault();
        if(downsampled_data.length === 0){
            http.get('/api/downsample', {params: {
                dataset: dataset
            }})
            .then((response) => {
                // console.log(response.data);
                // let downsampled_data: Array<Item> = []
                response.data.forEach((item: any) => {
                    downsampled_data.push({x: item.x, y: item.y});
                });
                setShowSampledData(true)
                // setDownSampling(!downSampling);
                // d3.select(".downsampled_signal").attr("d", line(downsampled_data) as string);
                
                console.log(downsampled_data);
    
            }).catch((error) => {
                console.log(error);
            });
        }
        else if(showSampledData){
            setShowSampledData(false);
        }
        else if(!showSampledData){
            setShowSampledData(true);
        }
    }

    function handleAleaClick(e: any){
        e.preventDefault();
        http.get('/api/alea')
        .then((response) => {
            // console.log(response.data)
            setAlea(response.data);
        }).catch((error) => {
            console.log(error);
        });
        
    }


    function handleAnomalyClick(e: any){
        e.preventDefault();

        http.get('/api/anomaly')
        .then((response) => {
            const anomaly_tag_height = 10;
            console.log(response.data);
            response.data.forEach((item: any) => {
                d3.select(".anomaly_tag").append("rect")
                    .attr("x", x(item.x1))
                    .attr("y", anomaly_tag_height)
                    .attr("width",  x(item.x2) - x(item.x1))
                    .attr("height", anomaly_tag_height)
                    .attr("fill", colorScale(item.anomaly_score));
                d3.select(".anomaly_box").append("rect")
                    .attr("x", x(item.x1))
                    .attr("y", anomaly_tag_height + anomaly_tag_height)
                    .attr("width",  x(item.x2) - x(item.x1))
                    .attr("height", height - marginBottom - 2 * anomaly_tag_height)
                    .attr("fill", "#ebebeb");
            });
        }).catch((error) => {
            console.log(error);
        });
    }

    const brush_check_color = ["green", "red", "blue"];

    function handleBrushClick_1(e: React.MouseEvent<HTMLDivElement>){
        e.preventDefault();

        d3.select(".rect_brush_1").append("rect")
            .attr("x", current_brush_loc.x0 )
            .attr("y", current_brush_loc.y0 )
            .attr("width", (current_brush_loc.x1 - current_brush_loc.x0))
            .attr("height", (current_brush_loc.y1 - current_brush_loc.y0))
            .attr("rx", 5)  
            .attr("ry", 5)
            .attr("fill", brush_check_color[current_brush_index])
            .attr("stroke", brush_check_color[current_brush_index])
            .attr("stroke-width", 1)
            .attr("stroke-opacity", 0.3)
            .attr("fill-opacity", 0.1); 

        selected_range1 = [x.invert(current_brush_loc.x0), x.invert(current_brush_loc.x1)];
        // console.log(selected_range)
        http.get('/api/mult_epis', {params: {
            x0: selected_range1[0],
            x1: selected_range1[1]
        }})
        .then((response) => {
            // console.log(response.data);
            // console.log(Object.values(ret))
            setMultiEpis1(response.data);
        }).catch((error) => {
            console.log(error);
        });

        // this code should be moved into the above `then` , but i ignore it yet.
        current_brush_index = 1;
        e.currentTarget.style.visibility = "hidden";

        d3.select(".brush_tooltip_cancel_1")
            .style("visibility", "visible")
            .style("left", tooltip_loc[0] + "px") // 112 is margin-left of div[#root]
            .style("top", tooltip_loc[1] +  "px")
            .style("z-index", 1001); // 1000 is the z-index of brush_tooltip

    }

    function handleBrushClick_2(e: React.MouseEvent<HTMLDivElement>){
        e.preventDefault();

        console.log(current_brush_loc)
        d3.select(".rect_brush_2").append("rect")
            .attr("x", current_brush_loc.x0 )
            .attr("y", current_brush_loc.y0 )
            .attr("width", (current_brush_loc.x1 - current_brush_loc.x0))
            .attr("height", (current_brush_loc.y1 - current_brush_loc.y0))
            .attr("rx", 5)  
            .attr("ry", 5)
            .attr("fill", brush_check_color[current_brush_index])
            .attr("stroke", brush_check_color[current_brush_index])
            .attr("stroke-width", 1)
            .attr("stroke-opacity", 0.3)
            .attr("fill-opacity", 0.1); 

        selected_range2 = [x.invert(current_brush_loc.x0), x.invert(current_brush_loc.x1)];
        // console.log(selected_range)
        http.get('/api/mult_epis', {params: {
            x0: selected_range2[0],
            x1: selected_range2[1]
        }})
        .then((response) => {
            // console.log(response.data);
            // const ret = response.data;
            setMultiEpis2(response.data)
            // console.log(Object.values(ret))
            // setMultiEpis2(Object.values(ret));

        }).catch((error) => {
            console.log(error);
        });
        // this code should be moved into the above `then` , but i ignore it yet.
        current_brush_index = 2;
        e.currentTarget.style.visibility = "hidden";
        d3.select(".brush_tooltip_cancel_2")
            .style("visibility", "visible")
            .style("left", tooltip_loc[0] + "px") // 112 is margin-left of div[#root]
            .style("top", tooltip_loc[1] + "px")
            .style("z-index", 1001); // 1000 is the z-index of brush_tooltip
    }

    function handleBrushClick_3(e: React.MouseEvent<HTMLDivElement>){
        e.preventDefault();

        console.log(current_brush_loc)
        d3.select(".rect_brush_3").append("rect")
            .attr("x", current_brush_loc.x0 )
            .attr("y", current_brush_loc.y0 )
            .attr("width", (current_brush_loc.x1 - current_brush_loc.x0))
            .attr("height", (current_brush_loc.y1 - current_brush_loc.y0))
            .attr("rx", 5)  
            .attr("ry", 5)
            .attr("fill", brush_check_color[2]) // "#e5efff")
            .attr("stroke", brush_check_color[2])
            .attr("stroke-width", 1)
            .attr("stroke-opacity", 0.3)
            .attr("fill-opacity", 0.1); 

        selected_range3 = [x.invert(current_brush_loc.x0), x.invert(current_brush_loc.x1)];
        // console.log(selected_range)
        http.get('/api/mult_epis', {params: {
            x0: selected_range3[0],
            x1: selected_range3[1]
        }})
        .then((response) => {
            // const ret = response.data;
            setMultiEpis3(response.data)
            // console.log(Object.values(ret))
            // setMultiEpis3(Object.values(ret));

        }).catch((error) => {
            console.log(error);
        });
        // this code should be moved into the above `then` , but i ignore it yet.

        current_brush_index = 3;
        e.currentTarget.style.visibility = "hidden";
        d3.select(".brush_tooltip_cancel_3")
            .style("visibility", "visible")
            .style("left", tooltip_loc[0] + "px") // 112 is margin-left of div[#root]
            .style("top", tooltip_loc[1] + "px")
            .style("z-index", 1001); // 1000 is the z-index of brush_tooltip
    }

    function handleBrushCancelClick_1(e: React.MouseEvent<HTMLDivElement>){
        e.preventDefault();
        d3.select(".rect_brush_1").select("rect").remove();
        current_brush_index = 0;

        d3.select(".brush_tooltip_cancel_1")
            .style("visibility", "hidden")
            .style("pointerEvents", "none")
            .style("z-index", 0); // 1000 is the z-index of brush_tooltip
        d3.select(".brush_tooltip_1")
            .style("visibility", "visible")
        // d3.select(".epis_50_anim1").remove()
        // d3.select(".epis_95_anim1").remove()
        // d3.select(".epis_anim_refline10").remove();
        // d3.select(".epis_anim_refline12").remove();
        d3.select(".multiples1").selectAll("*").remove()

    }

    function handleBrushCancelClick_2(e: React.MouseEvent<HTMLDivElement>){
        e.preventDefault();
        d3.select(".rect_brush_2").select("rect").remove();

        current_brush_index = 1;
        d3.select(".brush_tooltip_cancel_2")
            .style("visibility", "hidden")
            .style("pointerEvents", "none")
            .style("z-index", 0); // 1000 is the z-index of brush_tooltip
        d3.select(".brush_tooltip_2")
            .style("visibility", "visible")
        d3.select(".multiples2").selectAll("*").remove()

    }
    
    function handleBrushCancelClick_3(e: React.MouseEvent<HTMLDivElement>){
        e.preventDefault();
        d3.select(".rect_brush_3").select("rect").remove();
        current_brush_index = 2;

        d3.select(".brush_tooltip_cancel_3")
            .style("visibility", "hidden")
            .style("pointerEvents", "none")
            .style("z-index", 0); // 1000 is the z-index of brush_tooltip
        d3.select(".brush_tooltip_3")
            .style("visibility", "visible")
        d3.select(".multiples3").selectAll("*").remove()
    }


    return (
        <>

            {/* <Button onClick={handleMultEpisClick}> Show Epis Sensitivety  </Button> */}
            <Row style={{marginTop: '10px'}}>
                <Col span={22}>
                    <div className="legend">
                        <Legend />
                    </div>
                    <div className="container">

                        {alea.length !== 0 && <Aleatoric alea={alea} width={width} height={alea_view_height} />}
                        {epis.length !== 0 && <Epistemic epis={epis} width={width} height={epis_view_height} />}

                        <svg ref={ref_svg as any} width={width} height={height}>
                            <text x={marginLeft + 15} y={marginTop + 5} textAnchor="middle" fontSize="10" fill="black" > Value </text>
                            <text x={width - marginLeft + 10} y={height - marginBottom - 3} textAnchor="middle" fontSize="10" fill="black" > Time </text>
                        
                            <g ref={gx} transform={`translate(0, ${height - marginBottom})`} />
                            <g ref={gy} transform={`translate(${marginLeft}, 0)`} />
                            <g className="anomaly_tag"></g>
                            <g className="anomaly_box"></g>
                            
                            {!showSampledData && <path fill="none" stroke="currentColor" className="raw_signal" strokeWidth="1.5" d={line(data) as string} />}
                            {showSampledData && <path fill="none" stroke="gray" opacity={0.3} className="downsampled_signal" strokeWidth="1.5" d={line(downsampled_data) as string} />}

                            <path className="epis_50" fill="steelblue" opacity="0.7" />
                            <path className="epis_95" fill="steelblue" opacity="0.3" />
                            {/* <text x={width / 2} y={marginTop - 10} textAnchor="middle" fontSize="20" fill="black" > Raw Signal </text> */}
                            
                            <g className="ref_line"></g>
                            
                            <g className="rect_brush_1"></g>
                            <g className="rect_brush_2"></g>
                            <g className="rect_brush_3"></g>
                            
                        </svg>

                    {/* { multi_epis1.length > 0 && <Multiples mult_epis1={multi_epis1} mult_epis2={multi_epis2} mult_epis3={multi_epis3} />  } */}
                    </div>

                    { 
                    multi_epis1 && 
                    <div className="multipes">  
                        <div className="multipes-content">
                            <span style={{"fontSize": "10px","writingMode": "vertical-lr", "transform": "rotate(180deg)", "color": "gray"}}>Detail</span>
                            {<Epistemic_anim response={multi_epis1} selected_data={data.slice(selected_range1[0], selected_range1[1])} color="green" mykey="1"/> }
                            { multi_epis2 && <Epistemic_anim response={multi_epis2} selected_data={data.slice(selected_range2[0], selected_range2[1])} mykey="2" color="red"/> }
                            { multi_epis3 && <Epistemic_anim response={multi_epis3} selected_data={data.slice(selected_range3[0], selected_range3[1])} mykey="3" color="blue"/> }
                        </div>
                        <div style={{"fontSize": "10px", "color": "gray"}}>Time</div>
                    </div>
                    }
                </Col>
                <Col span={2} >
                    <div className="button_list">
                        <Button size={"small"} onClick={handleEpisClick}>Epistemic</Button>
                        <Button size={"small"} onClick={handleDownsampleClick}> { showSampledData ? "Upsampling": "Downsampling"} </Button>
                        <Button size={"small"} onClick={handleAleaClick}> Aleatoric </Button>
                        <Button size={"small"} onClick={handleAnomalyClick}> Anomaly </Button>
                    </div>
                </Col>
            </Row>



            <div className="brush_tooltip_1" onClick={handleBrushClick_1}><CheckOutlined /> </div>
            <div className="brush_tooltip_2" onClick={handleBrushClick_2}><CheckOutlined /> </div>
            <div className="brush_tooltip_3" onClick={handleBrushClick_3}><CheckOutlined /> </div>
            <div className="brush_tooltip_cancel_1" onClick={handleBrushCancelClick_1}><CloseOutlined /></div>
            <div className="brush_tooltip_cancel_2" onClick={handleBrushCancelClick_2}><CloseOutlined /></div>
            <div className="brush_tooltip_cancel_3" onClick={handleBrushCancelClick_3}><CloseOutlined /></div>
        </>
    )
  
}