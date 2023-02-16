const {DeckGL, ScatterplotLayer, PathLayer, PolygonLayer, BitmapLayer, SolidPolygonLayer} = deck;

function hexToRgb(hex) {
    var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return [parseInt(result[1], 16), parseInt(result[2], 16), parseInt(result[3], 16)]
}

class Webgl_Map{
    constructor(
        colormap2d,
        map_texture,
        bounding_box,
        tooltip_cols,
        color_scale,
        color_mapper,
        data){
        this.data = data;
        this.object_selected_data = data;
        this.timestep_selected_data = data;
        this.area_scale = d3.scaleLinear()
            .range([4, 15])
            .domain(d3.extent(this.data.map(d => d.area)));
        this.color_scale = color_scale;
        this.color_mapper = color_mapper;
        this.tooltip_cols = tooltip_cols;
        this.highlight_data = data;
        this.colormap2d = colormap2d;
        this.map_texture = map_texture;
        this.bounding_box = bounding_box;
    }

    /**
     * Function that starts the map plot.
     */
    create_plot(){
        document.getElementById("data_plot").innerHTML = "";
        var map_div = document.createElement("div");
        map_div.id = "map_plot";
        document.getElementById("data_plot").appendChild(map_div);

        this.deckgl = new deck.DeckGL({
            mapboxApiAccessToken: 'pk.eyJ1IjoiZ2lvdmFuaXZhbGRyaWdoaSIsImEiOiJja21rc3I1NzcwN3QxMnBvZGI4Nmk2b2N4In0.45WjGvP8-yh5Kl5FpBLb_g',
            mapStyle: 'mapbox://styles/mapbox/light-v9',
            container: "map_plot",
            initialViewState: {
                longitude: -59.896603,
                latitude: 31.357102,
                zoom: 1
            },
            controller: true,
            layers: []
        });
        
        var tooltip_div = document.createElement("div");
        tooltip_div.id = "map_tooltip";
        document.getElementById("data_plot").appendChild(tooltip_div);
    }

    set_highlight_data(highlight_data) {
        this.highlight_data = highlight_data;
    }

    set_cur_timestep(timestep) {
        this.cur_timestep = timestep;
        if (this.cur_timestep){
            this.timestep_selected_data = this.data.filter(d => d.timestep == timestep);
        } else {
            this.timestep_selected_data = this.data;
        }
        
        this.render();
    }

    set_selected_objects(selected_objects) {
        this.selected_objects = selected_objects;
        if (this.selected_objects.length > 0) {
            this.object_selected_data = this.data.filter(d => selected_objects.includes(d.object));
        } else {
            this.object_selected_data = this.data;
        }
        this.render();
    }

    /**
     * Function that draw all shapes in the map plot with selections.
     */
    render(){
        const self = this;
       
        var layers = [];

        //if(this.map_texture) 
        layers.push(self.draw_map_texture());
        
        //Scatter layer of background points
        const scatterLayer = new ScatterplotLayer({
            id: 'complete_layer',
            data: self.data,
            pickable: true,
            filled: true,
            opacity: 0.025,
            stroked: false,
            radiusScale: 1,
            radiusUnits: 'pixels',
            getRadius: 5,
            getPosition: d => [d.longitude, d.latitude],
            getFillColor: d => hexToRgb(self.color_scale(self.color_mapper(d))),
        })

        layers.push(scatterLayer);   

        layers.push(new PolygonLayer({
            id: "objects_shape",
            data: self.object_selected_data,
            filled: true,
            extruded: true,
            stroked: true,
            wireframe: true,
            opacity: 0.7,
            lineWidthMinPixels: 3,
            getElevation: (d, i) => i,
            getPolygon: d => d.points_coords,
            getFillColor: d =>  hexToRgb(self.color_scale(self.color_mapper(d))),
            getLineColor: [80, 80, 80],
            getLineWidth: 4
        }))

        layers.push(self.draw_path_layer());     
            
        self.deckgl.setProps({layers: layers});
    }

    draw_map_texture(){
        const self = this;
        var lon_min = this.bounding_box[0][0]; 
        var lat_min = this.bounding_box[0][1];
        var lon_max = this.bounding_box[1][0];
        var lat_max = this.bounding_box[1][1];
        const texture_layer = new BitmapLayer({
            id : 'image-layer',
            bounds : [lon_min, lat_min, lon_max, lat_max],
            image: 'http://127.0.0.1:5000/static/data/' + self.colormap2d + '.png',
            opacity: 0.4
    
        })
        return texture_layer;
    }

    generate_path(data_input){
        const self = this;
        return [].concat(...d3.groups(data_input, d => d.object)
            .map(d =>{
                var data = d[1];
                data.sort((a, b) => a.timestep - b.timestep);
                var res = [];
                for(let i = 1; i < data.length; i++) {
                    let d0 = data[i - 1];
                    let d1 = data[i];
                    res.push({"object": d[0],
                        "color_attr": self.color_mapper(d1),
                        "path": [
                            [d0.longitude, d0.latitude], 
                            [d1.longitude, d1.latitude]]})
                }
                return res;
            }));
    }

    draw_path_layer(){
        const self = this;
        //Layer of line connecting points in the time order
        var path_objects = self.generate_path(self.object_selected_data);

        const path_layer = new PathLayer({
            id: 'path-layer',
            data: path_objects,
            pickable: true,
            widthScale: 20,
            widthMinPixels: 3,
            getPath: d => d.path,
            getColor: d => hexToRgb(self.color_scale(d.color_attr)),
            getWidth: 10
        });

        return path_layer;

    }
}