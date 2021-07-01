<template>
    <div style="margin: 0">
        <div class="baseMap">
            <l-map
                    ref="map"
                    style="width: 100%; height: 100vh;"
                    :zoom="zoom"
            >
                <l-tile-layer :url="url" />
            </l-map>
        </div>
        <div class="box1">
            <div style="display: block">
                <div class="circle" style="background-color: #ff1120;" />
                true value
            </div>
            <div style="display: block">
                <div class="circle" style="background-color: #365bff;" />
                predicted value
            </div>
        </div>
        <div class="box2">
            <el-button style="float: left; margin-left: 0; margin-bottom: 2px" type="primary" size="mini" @click="showData('jakarta-04-id')">Jakarta</el-button>
            <el-button style="float: left; margin-left: 0; margin-bottom: 2px" type="primary" size="mini" @click="showData('city of london-enggla-gb')">London</el-button>
            <el-button style="float: left; margin-left: 0; margin-bottom: 2px" type="primary" size="mini" @click="showData('los angeles-ca037-us')">Los Angeles</el-button>
            <el-button style="float: left; margin-left: 0; margin-bottom: 2px" type="primary" size="mini" @click="showData('kuala lumpur-14-my')">Kuala Lumpur</el-button>
            <el-button style="float: left; margin-left: 0; margin-bottom: 2px" type="primary" size="mini" @click="showData('bandung-30-id')">Bandung</el-button>
            <el-button style="float: right" type="primary" size="mini" @click="getBaseData()">reset</el-button>
        </div>
    </div>
</template>
<script>
    import { LMap, LTileLayer } from 'vue2-leaflet'
    import L from 'leaflet'
    import mapData from './datalist'
    export default {
        name: 'Map',
        components: {
            LMap, LTileLayer
        },
        data() {
            return {
                url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
                zoom: 2,
                layer: '',
                layer1: '',
                map: '',
                data: [],
                users: 0,
            }
        },
        created() {
            this.$nextTick(()=>{
                this.init()
            })
        },
        methods:{
            init(){
                // console.log(data.datalist)
                this.map = this.$refs.map.mapObject
                this.layer = L.featureGroup([])
                this.layer1 = L.featureGroup([])
                this.map.addLayer(this.layer)
                this.map.addLayer(this.layer1)
                this.data = {}
                for(const item of mapData.datalist){
                    if(!Object.prototype.hasOwnProperty.call(this.data, item[3])){
                        this.data[item[3]] = {
                            coordinate: item[2],
                            predicted: [{
                                name: item[1],
                                coordinate: item[0],
                                distances: item[4],
                                acc: 1,
                                mean: 0
                            }],
                        }
                    }else {
                        // console.log('in', item[1])
                        this.data[item[3]].predicted.push({
                            name: item[1],
                            coordinate: item[0],
                            distances: item[4]
                        })
                    }
                }
                this.users = 0
                for(const item in this.data){
                    let count = 0
                    let sum = 0
                    if(Object.prototype.hasOwnProperty.call(this.data, item)) {
                        for (const pred of this.data[item].predicted) {
                            if (pred.name === item) {
                                count++;
                            }
                            sum += pred.distances
                        }
                        this.data[item].acc = count / item.length
                        this.data[item].mean = sum / item.length
                        this.users += item.length
                    }
                }
                console.log(this.data)
                console.log(this.users)
                this.getBaseData()
            },
            getBaseData(){
                this.layer.clearLayers()
                this.layer1.clearLayers()
                for (const item in this.data) {
                    if(Object.prototype.hasOwnProperty.call(this.data, item)) {
                        const content = `<div class="boundaryMask-popup">` +
                            `<p class="title">` + item + `</p>` +
                            `<ul class="info">` +
                            `<li><span>users:</span>` + this.data[item].predicted.length +`<span>/</span>` + this.users + `</li>` +
                            `<li><span>accuracy:</span>` + this.data[item].acc.toFixed(2) + `<span>%</span></li>` +
                            `<li><span>Median Error Distance:</span>` + this.data[item].mean.toFixed(2) + `<span>km</span></li>` +
                            `</ul>` +
                            `<div style="width:10px;height:10px" id="pptnMapChart"></div>` +
                            ` </div>`
                        let circle = L.circle(this.data[item].coordinate,
                            {
                                color: '#ff1120',
                                fillColor: '#ff8e83',
                                radius: this.data[item].mean,
                                fillOpacity: 0.5
                            }).bindPopup(content).addTo(this.layer)
                        circle.on('mouseover', function () {
                            this.openPopup();
                        })
                        var _this = this
                        circle.on('click', function () {
                            _this.layer.clearLayers()
                            _this.layer1.clearLayers()
                            _this.showData(item)
                        })
                    }
                }
            },
            showData(item){
                this.layer.clearLayers()
                this.layer1.clearLayers()
                const content = `<div class="boundaryMask-popup">` +
                    `<p class="title">`+ item +`</p>` +
                    `<ul class="info">` +
                    `<li><span>users:</span>` + this.data[item].predicted.length +`<span>/</span>` + this.users + `</li>` +
                    `<li><span>accuracy:</span>` + this.data[item].acc.toFixed(2) + `<span>%</span></li>` +
                    `<li><span>Median Error Distance:</span>`+ this.data[item].mean.toFixed(2) +`<span>km</span></li>` +
                    `</ul>` +
                    `<div style="width:10px;height:10px" id="pptnMapChart"></div>` +
                    ` </div>`
                for(const pred of this.data[item].predicted) {
                    L.polyline([this.data[item].coordinate, pred.coordinate], {
                        color: '#40ff5b',
                        weight: 2
                    }).addTo(this.layer1)
                    L.circle(pred.coordinate,
                        { color: '#365bff', fillColor: '#6686ff', radius: 1500, fillOpacity: 1 }).bindPopup(pred.name).addTo(this.layer1)
                    L.circle(this.data[item].coordinate,
                        { color: '#ff1120', fillColor: '#ff8e83', radius: this.data[item].mean, fillOpacity: 1 }).bindPopup(content).openPopup().addTo(this.layer)
                }
            }

        }
    }
</script>

<style scoped>
    .baseMap {
        width: 100%;
        height: 100%;
        position: relative;
        z-index: 0;
    }
    .box1 {
        padding: 5px;
        float: left;
        width: 10%;
        height: 6%;
        margin-top: -85vh;
        margin-left: 5%;
        position: relative;
        z-index: 1;
        background-color: rgba(44, 44, 44, 0.25);
    }
    .box2 {
        padding: 5px;
        float: right;
        width: 10%;
        height: 6%;
        margin-top: -85vh;
        margin-right: 5%;
        position: relative;
        z-index: 1;
        background-color: rgba(44, 44, 44, 0.25);
        font-size: 20px;
    }
    .circle {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 5px;
    }
</style>
