// antd
import { useState, useEffect } from 'react';
import {Card, Col, Row, Button, Dropdown, Space} from 'antd';

// route
import http from './https';

// style
import "./layout.css"

// custom components
import Test from './examples/Test2'
import MainChart from './MainChart';


import type { Item } from './MainChart';
import type { MenuProps } from 'antd';
let data: Array<Item> = [];
let dataset: string = "SMAP_P1";




export default function Layout() {
    const [isPlot, setIsPlot] = useState(false);

    const items: MenuProps['items'] = [
        {
          key: '1',
          label: (
            <a target="_blank" rel="noopener noreferrer" onClick={(e) => handleClick(e, setIsPlot)}>
              SMAP P1
            </a>
          ),
        },
        {
          key: '2',
          label: (
            <a target="_blank" rel="noopener noreferrer" onClick={(e) => handleClick(e, setIsPlot)}>
              NY Taxi
            </a>
          ),
        },
        {
          key: '3',
          label: (
            <a target="_blank" rel="noopener noreferrer" onClick={(e) => handleClick(e, setIsPlot)}>
              SWAT PIT502
            </a>
          ),
        },
        {
          key: '4',
          label: (
            <a target="_blank" rel="noopener noreferrer" onClick={(e) => handleClick(e, setIsPlot)}>
              Toy Example
            </a>
          ),
        },
      ];
    

    function handleClick(e: any, onIsPlotChange: (state: boolean) => void){
        e.preventDefault();
        console.log(e.target.innerText);
        
        // get raw signal data
        if (e.target.innerText === "SMAP P1") {
            dataset = "SMAP_P1"
        } else if (e.target.innerText === "NY Taxi") {
            dataset = "NY_Taxi"
        } else if(e.target.innerText === "SWAT PIT502") {
            dataset = "SWAT_PIT502"
        } else if(e.target.innerText === "Toy Example") {
            dataset = "toy_example";
        }
        http.get(`/api/${dataset}`)
        .then((response) => {
            // console.log(response.data);
            response.data.forEach((item: any) => {
                data.push({x: item.x, y: item.y});
            });
            onIsPlotChange(true);
        }).catch((error) => {
            console.log(error);
        });
    }



    return (
        <div className='layout'>
            <main>
                <Row>
                    <Col span={4} >
                        <span className='caption'>UATSAD</span>
                    </Col>
                    <Col span={2} offset={18}>
                    <Dropdown menu={{ items }} placement="bottomLeft">
                        <Button type='primary' size='small'>Load Dataset</Button>
                    </Dropdown>
                        {/* <Button type={'primary'} size='small' onClick={(e) => handleClick(e, setIsPlot)}>Load SMAP P1</Button> */}
                    </Col>
                </Row>
                <Row>
                    <Col span={24}>
                        {isPlot && <MainChart data={data} dataset={dataset}/>}
                    </Col>       
                    {/* <Col span={6}>
                        <Button onClick={(e) => handleClick(e, setIsPlot)}>Load SMAP P1</Button>
                        <Legend />
                    </Col> */}
                </Row>

            </main>
            <footer>

            </footer>
        </div>
    )
};

