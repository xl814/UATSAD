import * as d3 from "d3";
import { useRef, useEffect, useState } from "react";

import "./MainChart.css";


import Item from "antd/es/list/Item";

interface Item {
    x: number,
    y: number
}


interface MultiplesProps {
    mult_epis1: Array<Array<Item>>;
    mult_epis2: Array<Array<Item>>;
    mult_epis3: Array<Array<Item>>;
    width?: number;
    height?: number;
    marginTop?: number;
    marginRight?: number;
    marginBottom?: number;
    marginLeft?: number;
}
interface EpistemicProps {
    // mult_epis: Array<Array<Item>>;
    // mult_epis_50: Array<Array<Item>>;
    // mult_epis_95: Array<Array<Item>>;
    selected_data: Array<Item>;
    response: {epis: Array<Array<Item>>, epis_lower_50: Array<Array<Item>>, epis_upper_50: Array<Array<Item>>, epis_lower_95: Array<Array<Item>>, epis_upper_95: Array<Array<Item>>};
    color?: string;
    mykey?: string;
    width?: number;
    height?: number;
    marginTop?: number;
    marginRight?: number;
    marginBottom?: number;
    marginLeft?: number;
}

export default function Epistemic_anim({response, selected_data, color, mykey, width = 250, height = 150, marginTop = 5, marginRight = 5, marginBottom = 20, marginLeft = 40}: EpistemicProps){
    // let mult_epis: Array<Array<Item>> = Array(3);
    // mult_epis = response.epis;
    const mult_epis: Array<Array<Item>> = response.epis;
    const mult_epis_lower_50: Array<Array<Item>> = response.epis_lower_50;
    const mult_epis_upper_50: Array<Array<Item>> = response.epis_upper_50;
    const mult_epis_lower_95: Array<Array<Item>> = response.epis_lower_95;
    const mult_epis_upper_95: Array<Array<Item>> = response.epis_upper_95;
    

    let mult_epis_50: Array<Array<[number, number]>> = Array(3);
    let mult_epis_95: Array<Array<[number, number]>> = Array(3);
    for (let i = 0; i < mult_epis_lower_50.length; i++) {
        let epis_50: Array<[number, number]> = [];
        for (let j = 0; j < mult_epis_lower_50[i].length; j++) {
            epis_50.push([mult_epis_lower_50[i][j].y, mult_epis_upper_50[i][j].y]);
        }
        mult_epis_50[i] = epis_50;
    }

    for (let i = 0; i < mult_epis_lower_95.length; i++) {
        let epis_95: Array<[number, number]> = [];
        for (let j = 0; j < mult_epis_lower_50[i].length; j++) {
            epis_95.push([mult_epis_lower_95[i][j].y, mult_epis_upper_95[i][j].y]);
        }
        mult_epis_95[i] = epis_95;
    }
    
    const epis_upper_95_max = mult_epis_upper_95[0].map((item: Item, i) => Math.max(mult_epis_upper_95[0][i].y, mult_epis_upper_95[1][i].y, mult_epis_upper_95[2][i].y));
    const epis_upper_95_min = mult_epis_upper_95[0].map((item: Item, i) => Math.min(mult_epis_upper_95[0][i].y, mult_epis_upper_95[1][i].y, mult_epis_upper_95[2][i].y));
    const epis_lower_95_max = mult_epis_lower_95[0].map((item: Item, i) => Math.max(mult_epis_lower_95[0][i].y, mult_epis_lower_95[1][i].y, mult_epis_lower_95[2][i].y));
    const epis_lower_95_min = mult_epis_lower_95[0].map((item: Item, i) => Math.min(mult_epis_lower_95[0][i].y, mult_epis_lower_95[1][i].y, mult_epis_lower_95[2][i].y));
    const refline_upper_95_max: Array<Item> = [];
    const refline_upper_95_min: Array<Item> = [];
    const refline_lower_95_max: Array<Item> = [];
    const refline_lower_95_min: Array<Item> = [];

    mult_epis_upper_95[0].forEach((item: Item, i: number) => {
        refline_upper_95_max.push({x: item.x, y: epis_upper_95_max[i]});
        refline_upper_95_min.push({x: item.x, y: epis_upper_95_min[i]});
        refline_lower_95_max.push({x: item.x, y: epis_lower_95_max[i]});
        refline_lower_95_min.push({x: item.x, y: epis_lower_95_min[i]});
    });

    const data_begin = mult_epis[0][0].x; 
    const svg_ref = useRef<SVGSVGElement>(null);
    // const gx = useRef<null|SVGGElement>(null);
    // const gy = useRef<null|SVGGElement>(null);

    const x = d3.scaleLinear(d3.extent(mult_epis[0], (d: Item)  => d.x) as [number, number], [marginLeft, width - marginRight]);
    const y = d3.scaleLinear().domain([
        // d3.min(mult_epis_lower_95, (d: Array<Item>) => d3.min(d, (d: Item) => d.y)) as number,
        // epis_95_lower_min,
        // d3.max(mult_epis_upper_95, (d: Array<Item>) => d3.max(d, (d: Item) => d.y)) as number
        // epis_95_upper_max
        // -1.2, 1.5
        d3.min(selected_data, (d: Item) => d.y) as number - 0.2,
        d3.max(selected_data, (d: Item) => d.y) as number
        // 0.6
    ]).range([height - marginBottom, marginTop]);

    const line = d3.line((d: Item) => x(d.x), (d: Item) => y(d.y));
    const area = d3.area()
        .x((d, i) => x(i + data_begin))
        .y0((d) => y(d[0]))
        .y1((d) => y(d[1]));

    console.log(refline_lower_95_max, refline_lower_95_min, refline_upper_95_max, refline_upper_95_min)
    // useEffect(() => void d3.select(gx.current).call(d3.axisBottom(x).ticks(5) as any), [gx, x]);
    // useEffect(() => void d3.select(gy.current).call(d3.axisLeft(y).ticks(5) as any ), [gy, y]);
    useEffect(() => {
        console.log(svg_ref.current)
        const gx = d3.select(svg_ref.current).append("g").attr("transform", `translate(0, ${height - marginBottom})`) as any;
        const gy = d3.select(svg_ref.current).append("g").attr("transform", `translate(${marginLeft}, 0)`) as any;
        console.log(gx)
        gx.call(d3.axisBottom(x).ticks(5) as any);
        gy.call(d3.axisLeft(y).ticks(5) as any )
        // d3.select(gx.current).selectChild("path").attr("stroke", "gray");
        // d3.select(gy.current).selectChild("path").attr("stroke", "gray")
        const refline = [refline_lower_95_max, refline_lower_95_min, refline_upper_95_max, refline_upper_95_min]
        for (let i = 0; i < 4; i++)
            d3.select(svg_ref.current).append("path")
                .attr("fill", "none")
                .attr("stroke", "gray")
                .attr("stroke-width", 1)
                .attr("opacity", 0.3)
                .attr("stroke-dasharray", "2 2")
                .attr("class", "refline")
                .attr("d", line(refline[i]));
        
        


        // for(let i of [0, 2]){
        //     d3.select(svg_ref.current).append("path")
        //         .attr("fill", "none")
        //         .attr("stroke", "gray")
        //         .attr("stroke-width", 1)
        //         .attr("opacity", 0.3)
        //         .attr("stroke-dasharray", "2 2")
        //         .attr("class", "epis_anim_refline_lower" + mykey + i)
        //         .attr("d", line(mult_epis_lower_95[i]));
        // }

        d3.select(svg_ref.current).append("path")
                .attr("fill", "none")
                .attr("stroke", "gray")
                .attr("stroke-width", 1.5)
                .attr("opacity", 0.3)
                .attr("class", "selected_data" + mykey )
                .attr("d", line(selected_data));

        const epis_50_anim_path = d3.select(svg_ref.current).append("path")
            .attr("fill", "steelblue")
            .attr("class", "epis_50_anim" + mykey)
            .attr("opacity", "0.7")
            .attr("stroke", "steelblue")
            .attr("strokeWidth", "1")
            .attr("d", area(mult_epis_50[0]));

        const epis_95_anim_path = d3.select(svg_ref.current).append("path")
            .attr("fill", "steelblue")
            .attr("class", "epis_95_anim" + mykey)
            .attr("opacity", "0.3")
            .attr("stroke", "steelblue")
            .attr("strokeWidth", "1")
            .attr("d", area(mult_epis_95[0]));

        

        // play(area_mult_epis, 1);
        play(mult_epis_50, epis_50_anim_path, 1);
        play(mult_epis_95, epis_95_anim_path, 1);
        const svg_selection = d3.select(svg_ref.current);

        return ()=>{
            // d3.select(".epis_anim"+mykey).remove();
            // console.log(svg_ref.current)
            svg_selection.selectAll("*").remove();
            // d3.select(".epis_95_anim" + mykey).remove();
            // d3.select(".selected_data" + mykey).remove();
            // d3.selectAll('[class^="epis_anim_refline"]').remove();
            // d3.select(".epis_anim_refline" + mykey + 0).remove();
            // d3.select(".epis_anim_refline" + mykey + 2).remove();
        }
    }, [response])

    function play(area_epis: Array<[number,number][]> , path: any, i: number){
            if(i === 0) 
                i = 1;
            // console.log(i)
            path
            .transition()
                .duration(500)
                .attr("d", area(area_epis[i]) as any)
            .transition()
                .duration(500)
                .attr("d", area(area_epis[0]) as any)
            .on("end", ()=>{
                // console.log(i)
                play(area_epis, path, (i+1) % area_epis.length);
            })
            .end()
              
    } 




    return (
        <svg className={`multiples${mykey}`} ref={svg_ref} width={width} height={height} style={{"border": `0.5px solid ${color}`}}>
             {/* <g ref={gx} transform={`translate(0, ${height - marginBottom})`} />
             <g ref={gy} transform={`translate(${marginLeft}, 0)`} /> */}

             {/* <path className="anim_epis_50" fill="steelblue" opacity="0.7" />
             <path className="anim_epis_95" fill="steelblue" opacity="0.3" /> */}
             {/* <path fill="none" stroke="currentColor" className="raw_signal" strokeWidth="1.5" d={line(selected_signal) as string} /> */}
             {/* <path className={"epis_anim" + mykey} fill={color} opacity="0.5" stroke="black" strokeWidth={1}  d={area(area_mult_epis[0]) as string} /> */}
        </svg>
    )
}


// export default function Multiples({mult_epis1, mult_epis2, mult_epis3, width = 900, height = 100, marginTop = 3, marginRight = 20, marginBottom = 5, marginLeft = 40}: MultiplesProps){
    


//     return (
//         <div className="multipes">  
//             <div className="multipes-content">
//                 <span style={{"fontSize": "10px","writingMode": "vertical-lr", "transform": "rotate(180deg)"}}>Epistemic</span>
//                 <Epistemic_anim mult_epis={mult_epis1} color="green" mykey="1"/>
//                 { mult_epis2.length !== 0 && <Epistemic_anim mult_epis={mult_epis2} mykey="2" color="red"/> }
//                 { mult_epis3.length !== 0 && <Epistemic_anim mult_epis={mult_epis3} mykey="3" color="blue"/> }
//             </div>
//             <div style={{"fontSize": "10px"}}>Time</div>
//         </div>

//     )
// }