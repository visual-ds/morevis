function componentToHex(c) {
    var hex = c.toString(16);
    return hex.length == 1 ? "0" + hex : hex;
}

function rgbToHex(r, g, b) {
    return "#" + componentToHex(r) + componentToHex(g) + componentToHex(b);
}

class Visualization {
    constructor() {
        //Datasets configurations
        this.start_dataset_config();
        this.data = new Object();

        // Main plot visual configuration
        this.margin = {"left": 40, "top": 10, "bottom": 40, "right":10};
        this.colorBar_width = 25;
        this.area_error_spurious_height = 30;
        this.width = 870 - this.margin.left - this.margin.right - this.colorBar_width;
        this.height = 390 - this.margin.top - this.margin.bottom - this.area_error_spurious_height;
        this.svg_plot = d3.select("#morevis_svg")
            .attr("width", this.width + this.margin.left + this.margin.right + this.colorBar_width)
            .attr("height", this.height + this.margin.top + this.margin.bottom + this.area_error_spurious_height);
        this.intersection_svg = d3.select("#intersection_svg")
            .attr("width", 280)
            .attr("height", 140);
        this.color_legend_svg = d3.select("#color_legend")
            .append("svg")
            .attr("width", 250)
            .attr("height", 50);
        this.color_categoric_pallete = {
            "set3": [ '#a6cee3', '#1f78b4', '#e31a1c', '#6a3d9a', '#b15928', '#f2d933', '#ff7f00', '#cab2d6', '#fb9a99', '#b2df8a', '#fdbf6f', '#33a02c'],
            "paired":['#cab2d6', '#b2df8a', '#b15928', '#ffff99', '#6a3d9a', '#a6cee3', '#ff7f00', '#fb9a99', '#1f78b4', '#fdbf6f', '#e31a1c', '#33a02c']};

        this.color_continuous_pallete = [
            '#ffffe5','#f7fcb9','#d9f0a3',
            '#addd8e','#78c679','#41ab5d',
            '#238443','#006837','#004529'];
        this.color_continuous_pallete_multi_hue = ['#c7e9b4','#7fcdbb','#41b6c4','#1d91c0','#225ea8','#253494','#081d58'];
        //[
        //    '#440154', '#482475', '#414487', 
        //    '#355f8d', '#2a788e', '#21918c', 
        //    '#22a884', '#44bf70', '#7ad151'];
        this.color_scale;
        this.interactive_selections = {"object_clicked" : [], "time_selection": [], "space_selection": []};
        this.selected_timesteps = [];
        this.plot_colormap = false;
        
        return;
    }

    /**
     * Function that set all the attributes for each of the datasets 
     * that will be used in the visualization.
     */
     start_dataset_config(){
        this.dataset_config = {};

        this.dataset_config["wildtrack_small"] = {
            "type": "video", "width": 384, "height": 288, "xrange": 1920, "yrange": 1280,
            "rollup_func": function(v){
                return {
                    area_max : d3.max(v.map(d => d.area)),
                    area_std : d3.deviation(v.map(d => d.area)),
                    duration : d3.max(v.map(d => d.timestep)) - d3.min(v.map(d => d.timestep)),
                    x_std: d3.deviation(v.map(d => d.xcenter)),
                    y_std : d3.deviation(v.map(d => d.ycenter)) 
                };
            },
            "parallel_attr": [
                {"name": "area_max", "label": "Max Area"},
                {"name": "area_std", "label" : "Std. Area"},
                {"name": "duration", "label": "Duration"},
                {"name": "x_std", "label" : "Std. X Position"},
                {"name": "y_std", "label" : "Std. Y Position"}
            ],
            "time_interval": 5,
            "xticks": x => 5*x,
            "frame_map" : x => 5*x,
            "image_path": i => "static/data/c3_small/" + ("00000000" + (i)).slice(-8)  + ".png",
            "color_attr" : [{"value": "object", "label": "Object", "type": "categorical"}],
        };
        
        this.dataset_config["wildtrack_small_2"] = {
            "type": "video", "width": 384, "height": 288, "xrange": 1920, "yrange": 1280,
            "rollup_func": function(v){
                return {
                    area_max : d3.max(v.map(d => d.area)),
                    area_std : d3.deviation(v.map(d => d.area)),
                    duration : d3.max(v.map(d => d.timestep)) - d3.min(v.map(d => d.timestep)),
                    x_std: d3.deviation(v.map(d => d.xcenter)),
                    y_std : d3.deviation(v.map(d => d.ycenter)) 
                };
            },
            "parallel_attr": [
                {"name": "area_max", "label": "Max Area"},
                {"name": "area_std", "label" : "Std. Area"},
                {"name": "duration", "label": "Duration"},
                {"name": "x_std", "label" : "Std. X Position"},
                {"name": "y_std", "label" : "Std. Y Position"}
            ],
            "time_interval": 5,
            "xticks": x => 5*x,
            "frame_map" : x => 5*x,
            "image_path": i => "static/data/c3_small/" + ("00000000" + (i)).slice(-8)  + ".png",
            "color_attr" : [{"value": "object", "label": "Object", "type": "categorical"}],
        };

        this.dataset_config["hurdat"] = {
            "type": "map",
            "tooltip_cols": ["area", "timestep", "name", "pressure", "wind"],
            "rollup_func": function(v){
                const start_time = d3.min(v.map(d => d.timestep));
                const end_time = d3.max(v.map(d => d.timestep));
                const duration = (end_time - start_time);
                const start_longitude = v.filter(d => d.timestep == start_time)[0].xcenter;
                const start_latitude = v.filter(d => d.timestep == start_time)[0].ycenter;
                return {area_max : d3.max(v.map(d => d.area)),
                    wind_max: d3.max(v.map(d => d.wind)),
                    pressure_max: d3.max(v.map(d => d.pressure)),
                    duration: duration,
                    start_longitude: start_longitude,
                    start_latitude: start_latitude,
                    };
            },
            "parallel_attr": [
                {"name": "wind_max", "label": "Max Wind"},
                {"name": "pressure_max", "label": "Max Pressure"},
                {"name": "area_max", "label": "Max Area"},
                {"name": "duration", "label": "Duration"},
                {"name": "start_longitude", "label": "Start Lon."},
                {"name": "start_latitude", "label": "Start Lat."  },
            ],
            "xticks": x => {
                var i = (86400 * 2 * x) * 1000;
                var d = new Date(i);
                return d.getDate() + "/" + (d.getMonth() + 1);
            },
            "color_attr" : [
                {"value": "object", "label": "Object", "type": "categorical"},
                {"value": "wind", "label": "Wind", "type": "continuous"},
                {"value": "pressure", "label": "Pressure", "type": "continuous"},
                {"value": "latitude_start", "label": "Latitude (start)", "type": "continuous"},
                {"value": "longitude_start", "label": "Longitude (start)", "type": "continuous"},
                {"value": "area", "label": "Area", "type": "continuous"},
            ],
        }; 
    }

    /**
     * Update options for selector of curve color. 
     */
    update_curve_color_selector() {
        var options = "";
        const dataset = document.getElementById("dataset").value;
        this.dataset_config[dataset].color_attr.forEach(d => {
            options += `<option value ="${d.value}"> ${d.label} </option> `; 
        });
        document.getElementById("curve_color").innerHTML = options;
    }

    /**
     * Function that read the visualization options and update variables.
     */
    update_options(){
        this.fit_grid = true;
        this.dataset_name = document.getElementById("dataset").value;
        this.spurious_crossing_mark = document.getElementById("spurious_crossing_mark").value;
        this.intersection_view = "vertical-layout";//document.getElementById("intersection_view").value;
        this.space_break = false; //document.getElementById("space_break").checked;
        this.dataset_config_cur = this.dataset_config[this.dataset_name];
        this.colormap = document.getElementById("colormap").value
        this.time_distortion = false; //document.getElementById("time_distortion").checked;
        this.color_scale_name = document.getElementById("color_scale").value;
    }

     /**
     * Function that look the color scale selection on the form and set the d3 color scale
     * with new range and domain.
     */
    update_color_scale(){
        //console.log("entrou na função color_scale");
        const self = this;
        this.color_attr = document.getElementById("curve_color").value;
        const color_attr_info = self.dataset_config_cur.color_attr
            .filter(d => d.value == self.color_attr)[0];
        
        this.color_attr_info = color_attr_info;

        if(color_attr_info.type == 'categorical'){
            self.color_scale = d3.scaleOrdinal()
                .range(self.color_categoric_pallete[self.color_scale_name])
                .domain(self.data.original.map(d => d[self.color_attr]));
        }else if(color_attr_info.type == 'continuous'){
            self.color_scale = d3.scaleQuantile()
                .range(self.color_continuous_pallete_multi_hue)
                .domain(self.data.original.map(d => d[self.color_attr]));
        }
        //console.log(self.color_scale.domain())
    }

    /**
     * Remove the svg plot.
     */
    remove_plot(){
    this.interactive_selections = {"object_clicked" : [], "time_selection": [], "space_selection": []};
    if(document.getElementById("waiting")){
        document.getElementById("waiting").outerHTML = "";
    }
    this.intersection_svg.selectAll("*").remove();
    this.svg_plot.selectAll("*").remove();
    }

    update_handler(){
        this.remove_plot();
        var waiting_div = document.createElement("div");
        waiting_div.id = "waiting";
        waiting_div.innerHTML = "Loading...";
        document.getElementById("bottom_left").appendChild(waiting_div);
        this.update_options();
        this.set_dataset();
    }

    /**
     * Function that set the selected dataset and make request to server.
     */
    set_dataset(){
        var dataset = this.dataset_name;
        const type = this.dataset_config_cur.type;

        var ajax_request = $.ajax({
            type: 'POST',
            contentType: "application/json;charset=utf-8",
            url: "set_dataset",
            traditional: "true",
            data: JSON.stringify({dataset}),
            dataType: "json"
        })

        const self = this;
        ajax_request.always(function(html){
            
            d3.select("#parallel_plot").selectAll("*").remove();
            self.parallelCoordinates = new ParallelCoordinates(
                self,
                "#parallel_plot",
                self.dataset_config_cur.parallel_attr 
            );

            fetch("get_original_data")
            .then(response => {
                if(response.status == 200){
                    return response.json()
                }else{
                    throw new Error("Server error")
                }   
            }).then(function(response){

                self.data.original = response["objects"];
                self.data.intersections = response["intersections"];
                console.log(self.data.intersections)
                self.data.filtered = self.data.original.slice();
                self.update_color_scale();
                const rollup_func = self.dataset_config_cur.rollup_func;
                self.data.grouped = d3.rollups(self.data.original, rollup_func, d => d.object)
                .map(d => {
                    return {...{"object": d[0]}, ...d[1]};
                });

                self.parallelCoordinates.draw();
                
                if(self.dataset_config_cur.type == "map"){
                    var lats = self.data.original.map(d => d.latitude);
                    var lons = self.data.original.map(d => d.longitude);
                    var bounding_box = [[d3.min(lons), d3.min(lats)], 
                        [d3.max(lons), d3.max(lats)]];
                    self.webgl_map = new Webgl_Map(
                        self.colormap,
                        false,
                        bounding_box,
                        self.dataset_config_cur.tooltip_cols,
                        self.color_scale,
                        d => d[self.color_attr],
                        self.data.original
                    );
                    self.webgl_map.create_plot();
                }else {
                    console.log(self.dataset_config_cur["image_path"](0));
                    self.video_player = new VideoPlayer(
                        384,
                        288,
                        self.data.original,
                        self.dataset_config_cur["xrange"],
                        self.dataset_config_cur["yrange"],
                        self.dataset_config_cur["frame_map"],
                        self.dataset_config_cur["image_path"],
                        d => self.color_scale(d[self.color_attr]),
                        self.colormap
                    );
                }
                self.preprocess();
            })
        })
    }

    /**
     * Function that recieve an array of objects IDs and sent request to filter data to server.
     * @param {Array} selected_objects 
     */
     set_filtered_data(selected_objects){
        const self = this;
        var ajax_request = $.ajax({
            type: 'POST',
            contentType: "application/json;charset=utf-8",
            url: "/set_filtered_data",
            traditional: "true",
            data: JSON.stringify({selected_objects}),
            dataType: "json"
        });

        ajax_request.always(function(html){
            self.data.filtered = self.data.original.filter(d => selected_objects.includes(d.object));
            //self.video_player.update_data(self.data.filtered);
            self.preprocess();
        })
    }

    /**
     * Calls the preprocess algorithm to the server (projection and fix intersections).
     */
    preprocess(){
        const self = this;
        var myUrl = ('http://127.0.0.1:5000/get_scatter_data');
        fetch(myUrl)
        .then(response => {
            if(response.status == 200){
                return response.json()
            }else{
                throw new Error("Server error")
            }
        }).then(function(response){
            self.data.scatter = response["objects"];
            self.data.intersections = response["intersections"];
            self.data.scatter = self.data.scatter.sort(function(a, b){ return a.time - b.time; });
            if(document.getElementById("waiting")){
                document.getElementById("waiting").outerHTML = "";
            }
            self.update_color_scale(); 
            self.remove_plot();  
            //console.log("criou escala de cor e remove plot")
            if (self.dataset_config_cur.type == "map"){
                self.webgl_map.render();
            }
            if (document.getElementById("reset_zoom")) {
                //document.getElementById("highlight_intersections").outerHTML = "";
                document.getElementById("reset_zoom").outerHTML = "";
                document.getElementById("activate_selection").outerHTML = "";
            }

            //var highlight_intersections_button = document.createElement("button");
            //highlight_intersections_button.innerHTML = "Highlight Intersections";
            //highlight_intersections_button.id = "highlight_intersections";
            //self.highlight_intersections_active = false;
            //document.getElementById("morevis_buttons").appendChild(highlight_intersections_button);

            var reset_button = document.createElement("button");
            reset_button.innerHTML = "Reset Zoom";
            reset_button.id = "reset_zoom";
            document.getElementById("morevis_buttons").appendChild(reset_button);

            var selection_button = document.createElement("button");
            selection_button.innerHTML = "Activate Selection";
            selection_button.id = "activate_selection";
            document.getElementById("morevis_buttons").appendChild(selection_button);

            self.plot();
        }) 
    }

    /**
     * Function that draw the complete x-axis and then draw the guide lines for time distortion.
     * @param {*} g group object to draw the guide lines.
     * @param {*} x scale.
     * @param {*} vspace values for positioning the guide lines.
     */
    plot_grid = (g, x) => {
        const self = this;
        var y0 = 0;
        var y1 = this.height;
        
        //Plotting a gray background
        
        g.append("rect")
            .attr("fill", "#fff")//"#eee")
            .attr("width", this.width)
            .attr("height", this.height)
            .attr("x", 0)
            .attr("y", 0)
            .style("stroke", "#000000")
            .style("stroke-width", 0.5);

        g
            .append("defs")
            .append("svg:clipPath")
            .attr("id", "clip-x-grid")
            .append("svg:rect")
            .attr("id", "clip-x-grid-rect")
            .attr("x", 0)
            .attr("y", 0)
            .attr("width", this.width)
            .attr("height", this.height);

        var g_grid = g.append("g")
            .attr("clip-path", "url(#clip-x-grid)");

        //We'll verifiy the number of grid cells, 
        //if it is bigger than n_grids we will select n_grids major grids
        var xmax = Math.ceil(x.domain()[1]);
        var n_grids = Math.min(50, xmax);
        var step = Math.floor(xmax/n_grids);
        var i = Math.floor(x.domain()[0]);
        var line_grid = (x, i) => d3.line()([[x(i), y0], [x(i), y1]])

        while(i < xmax){
            g_grid.append("path")
                .data([i])
                .attr("d", d => line_grid(x, d))
                .attr("stroke", "#eee")//"#ffffff")
                .attr("stroke-opacity", 1)
                .attr("stroke-width", 1)
                .attr("fill", "none")
                .attr("class", "grid-divisor");
            i = i + step;
        }
        return line_grid;
    }

    plot_x_axis = (g, x) => {
        const self = this;
        g
            .append("defs")
            .append("svg:clipPath")
            .attr("id", "clip-x-axis")
            .append("svg:rect")
            .attr("id", "clip-x-axis-rect")
            .attr("x", 0)
            .attr("y", 0)
            .attr("width", this.width)
            .attr("height", 30);

        var g_objects = g.append("g")
            .attr("clip-path", "url(#clip-x-axis)");

        var xAxis = (g) => g.call(d3.axisBottom(x).tickFormat(self.dataset_config_cur.xticks));
        //.tickFormat(d => self.dataset_config_cur.xticks(d)));

        g_objects.append("g")
            .attr("class", "x-axis")
            .call(xAxis);
        return xAxis;
    }

    /**
     * Function that draw the y-axis and the breaks, also adds the brush selection.
     * @param {*} central_g group to add the selection rectangle
     * @param {*} g group object to add the axis
     * @param {*} y scale
     */
    plot_y_axis = (g, y) => {
        const self = this;

        g
            .append("defs")
            .append("svg:clipPath")
            .attr("id", "clip-y-axis")
            .append("svg:rect")
            .attr("id", "clip-y-axis-rect")
            .attr("x", -30)
            .attr("y", 0)
            .attr("width", 30)
            .attr("height", this.height);

        var g_objects = g.append("g")
            .attr("clip-path", "url(#clip-y-axis)");

        var yAxis = (g) => g.call(d3.axisLeft(y));

        g_objects.append("g")
            .attr("class", "y-axis")
            .call(yAxis);
        
        return yAxis;
    }

    /**
     * Create a bar plot over the timesteps
     * @param {group} g 
     * @param {scale} x 
     */
    plot_error_spurious = (g, x) => {
        const self = this;

        // Scales for area plot of spurious intersections by timestep
        var spurious_intersections = d3.groups(
            this.data.intersections, 
            d => d.timestep
        ).map(function(d) {
            return {
                "timestep": d[0],
                "total_intersections" : d3.sum(d[1].map(e => e.spurious_intersection)), 
                //"total_intersections" : d3.sum(d[1].map(e => e.intersection_1d & !e.spurious_intersection)), 
                "total_spurious": d3.sum(d[1].map(e => e.spurious_intersection)),
                "shape_type": d[1][0].shape_type
            }
        });

        g.append("defs")
            .append("svg:clipPath")
            .attr("id", "clip_main_error_area")
            .append("svg:rect")
            .attr("id", "clip_main_error_area_rect")
            .attr("x", 0)
            .attr("y", 0)
            .attr("width", this.width)
            .attr("height", this.area_error_spurious_height);

        // Draw text, background and axis
        g.append("text")
            .attr("text-anchor", "start")
            .attr("x", 0)
            .attr("y", -3)
            .attr("fill", "black")
            .text("Nº Spurious Intersections");

        g.append("rect")
            .attr("x", 0)
            .attr("y", 0)
            .attr("width", this.width)
            .attr("height", this.area_error_spurious_height - 5)
            .attr("stroke", "#bbbbbb")
            .attr("stroke-width", 0.5)
            .attr("fill", "#eeeeee");

        var max_intersections = d3.max(spurious_intersections.map(d => d.total_intersections));
        max_intersections = max_intersections > 0 ? max_intersections : 1;

        var y = d3.scaleLinear()
            .range([this.area_error_spurious_height - 5, 0])
            .domain([0, max_intersections]);

        g.call(d3.axisLeft(y)
            .tickValues([0, max_intersections])
            .tickFormat(d => parseInt(d)));

        var g_objects = g.append("g")
            .attr("clip-path", "url(#clip_main_error_area)");

        var tooltip = d3.select("#spurious_plot_tooltip");

        // Plot all intersections
        var bar_plot_intersections = g_objects.selectAll(".bar_plot_intersections")
        //    .data(spurious_intersections)
        //    .enter()
        //    .append("rect")
        //    .attr("class", "bar_plot_intersections")
        //    .attr("x", d => x(d.timestep + 0.25))
        //    .attr("y", d => y(d.total_intersections))
        //    .attr("width", x(0.5) - x(0))
        //    .attr("height", d => y(0) - y(d.total_intersections))
        //    .attr("fill", "#6495ed");

        var bar_plot_spurious = g_objects.selectAll(".bar_plot_spurious")
            .data(spurious_intersections)
            .enter()
            .append("rect")
            .attr("class", "bar_plot_spurious")
            .attr("x", d => x(d.timestep))
            .attr("y", d => y(d.total_spurious))
            .attr("width", x(1) - x(0))
            .attr("height", d => y(0) - y(d.total_spurious))
            .attr("fill", "#922B21")
            .attr("stroke", "none")
            .attr("stroke-width", 2)
            //.attr("opacity", d => d.shape_type == "rect" ? 1 : 0.4)
            //.on("click", click_handler)
            .on("mouseover", mouseover)
            .on("mousemove", mousemove)
            .on("mouseleave", mouseleave)
            
        function mousemove(event, d) {
            var xpos = event.layerX + 5;
            if (xpos > 750){
                xpos = xpos - d3.select("#spurious_plot_tooltip_svg").attr("width") - 20;
            }
            tooltip
            .style("left", xpos + "px")
            .style("top",  (event.layerY - 10) + "px");
        }

        function mouseleave() {
            tooltip.style("display", "none")
            d3.select(this).attr("stroke", "none");
            var intersection_svg = d3.select("#spurious_plot_tooltip_svg")
                .attr("width", 0)
                .attr("height", 0);
            intersection_svg.selectAll("*").remove();
            d3.selectAll(".object_area")
                .attr("class", d => "object_area object_area_deselected object_area_" + d.object)
                .attr("opacity", 1.0);
        }
       
        function mouseover(event, d) {
            tooltip.style("display", "block");
            d3.select(this).attr("stroke", "black");

            var selection = d3.select(this);
            var d = selection.data()[0];
            var timestep = d.timestep;
            var spurious_timestep = self.data.intersections.filter(d => {
                return d.timestep == timestep & d.spurious_intersection == 1;
            });

            console.log(spurious_timestep)

            // Highlight objects of the selected timestep
            // Change class of all objects to deselected
            d3.selectAll(".object_area")
                .attr("class", d => "object_area object_area_deselected object_area" + d.object)
                .attr("opacity", 0.1);
            self.update_data_selection();

            // Than change the opacity of the data of this spurious intersection 
            spurious_timestep.forEach(function(e) {
                d3.selectAll("#object_area_" + e.object1 + "_" + e.timestep.toString().replace(".", "_"))
                .attr("opacity", 1.0);
                d3.selectAll("#object_area_" + e.object2 + "_" + e.timestep.toString().replace(".", "_"))
                .attr("opacity", 1.0);
            });

            var plot_width = Math.min(120, spurious_timestep.length * 20) + 50;

            var intersection_svg = d3.select("#spurious_plot_tooltip_svg")
                .attr("width", plot_width)
                .attr("height", plot_width);

            intersection_svg.selectAll("*").remove();

            console.log(intersection_svg)

            const link_scale = d3.scaleLinear()
                .domain(d3.extent(spurious_timestep.map(d => d.area_1d)))
                .range([2.5, 6]);
            
            var network_data = spurious_timestep.slice();

            network_data.nodes = [...new Set(network_data
                .map(d => d.object1)
                .concat(network_data.map(d=> d.object2)))]
                .map(function(d){ 
                    return {
                        id: d, 
                        name: d, 
                        "color_attr": d[self.color_attr],
                    }
                });

          

            network_data.nodes = network_data.nodes.sort((a, b) => a.y_plot > b.y_plot).reverse();
            
            network_data.links = network_data
                .filter(d => !((d.value == 0) & (d.spurious == 0) | (d.object1 == d.object2)))
                .map(function(d){ 
                    return {"source" : d.object1, "target" : d.object2, "value" : d.area_1d, "spurious": d.spurious_intersection}
                });
            
            var net = new Network(
                network_data,
                self.dataset_config_cur.xticks(network_data[0].timestep),
                self.color_scale,
                link_scale,
                d => d.id,
                plot_width,
                plot_width,
                intersection_svg.append("g"),
                d3.select("#spurious_plot_tooltip_inside_tooltip"),
                false
            );

            net.plot_vertical();
        }

        return [bar_plot_intersections, bar_plot_spurious];
    }
    
    /**
     * Update the intersection plot.
     * @param {*} x D3 scale 
     * @param {*} y D3 scale
     */
    
    update_intersection_plot(x, y){
        const self = this;
        const [y0, y1] = this.selected_y_range;

        var data_intersection_selected = this.data.intersections
            .filter(d => {
                return (
                    (self.selected_timesteps.includes(d.timestep)) &
                    (d.y_bottom >= y0) & (d.y_top <= y1) &
                    (d.shape_type == "rect")
                )
            });
        
        var intersection_svg = d3.select("#intersection_svg")
            .attr("width", 0)
            .attr("height", 140)

        intersection_svg.selectAll("*").remove();
        

        var node_mapper = {};
        data_intersection_selected.forEach(d => {
            //console.log(d.object1_y, d.object2_y)
            node_mapper[d.object1] = {x : 70, y: 70, y_plot: d.object1_y};
            node_mapper[d.object2] = {x : 70, y: 70, y_plot: d.object2_y};
        });

        const link_scale = d3.scaleLinear()
            .domain(d3.extent(data_intersection_selected.map(d => d.area_1d)))
            .range([2.5, 6]);
        
        var optim_pos = true;
        this.selected_timesteps.forEach((timestep, i) => {
            
            var network_data = data_intersection_selected
                .filter(e => e.timestep == timestep);
            
            var node_colors = self.data.scatter
                .filter(d => d.timestep == timestep)
                .map(d => [d.object, d[self.color_attr]]);
            node_colors = new Map(node_colors);

            if (network_data.length > 0) {
                network_data.nodes = [...new Set(network_data
                    .map(d => d.object1)
                    .concat(network_data.map(d=> d.object2)))]
                    .map(function(d){ 
                        return {
                            id: d, 
                            name: d, 
                            "color_attr": node_colors.get(d),
                            x : node_mapper[d].x , 
                            y: node_mapper[d].y,
                            y_plot: node_mapper[d].y_plot
                        }
                    });
                    
                network_data.nodes = network_data.nodes.sort((a, b) => a.y_plot > b.y_plot).reverse();
                
                network_data.links = network_data
                    .filter(d => !((d.value == 0) & (d.spurious == 0) | (d.object1 == d.object2)))
                    .map(function(d){ 
                        return {"source" : d.object1, "target" : d.object2, "value" : d.area_1d, "spurious": d.spurious_intersection}
                    });
                
                // Calculate width of new plot
                var width_plot = Math.min(140, 50 + (network_data.links.length * 10));
                var height_plot = 140;
                var previous_width = parseInt(intersection_svg.attr("width"));

                intersection_svg
                    .attr("width", previous_width + width_plot);

                var g = intersection_svg.append("g")
                    .attr("id", "timestep_"+timestep)
                    .attr("transform", `translate(${previous_width}, 0)`)
                
                console.log(network_data.nodes)
                var net = new Network(
                    network_data,
                    self.dataset_config_cur.xticks(network_data[0].timestep),
                    self.color_scale,
                    link_scale,
                    d => d.color_attr,
                    width_plot,
                    height_plot,
                    g,
                    d3.select("#intersection_view_tooltip"),
                    optim_pos
                );

                optim_pos = false;

                if (self.intersection_view == "node_link") {
                    net.plot();
                    network_data.nodes.forEach(d => {
                        node_mapper[d.id]["x"] = d.x;
                        node_mapper[d.id]["y"] = d.y;
                    });
                } else {
                    net.plot_vertical();
                }
            }
        });
    }

    /**
     * Recieve the g element, the x/y scales and plot the shapes that represents objects.
     * @param {g} g group object to draw the shapes
     * @param {distortionScale} x scale
     * @param {breakScale} y scale
     */
    plot_objects_shapes = (g, x, y) => {
        const self = this;

        // Clip path for zooming
        var g_objects = g.append("g")
            .style("isolation", "isolate")
            .attr("clip-path", "url(#clip-main)");

        g
            .append("defs")
            .append("svg:clipPath")
            .attr("id", "clip-main")
            .append("svg:rect")
            .attr("id", "clip-rect")
            .attr("x", 0)
            .attr("y", 0)
            .attr("width", this.width)
            .attr("height", this.height);

        var tooltip = d3.select("#main_plot_tooltip");

        var mouseover = function(d) {
            tooltip.style("display", "block");
            //d3.select(this).attr("stroke", "black");
        }

        var mousemove = function(event, d) {
            tooltip
                .style("left", (event.layerX + 5) + "px")
                .style("top",  (event.layerY - 20) + "px")
                .html(`Object: ${d.object}<br>
                Timestep: ${parseInt(d.timestep)}
                `);
            self.interactive_selections.time_selection = d.timestep;
            if (self.dataset_config_cur.type == "video"){
                self.video_player.set_cur_frame(d.frame);
            } else if (self.dataset_config_cur.type == "map"){
                self.webgl_map.set_cur_timestep(parseInt(d.timestep));
            }
        }

        var mouseleave = function(d) {
            tooltip.style("display", "none")
            //d3.select(this)
            //.attr("stroke", d => self.color_scale(d[self.color_attr]))
            if (self.dataset_config_cur.type == "map"){
                self.webgl_map.set_cur_timestep();
            }
        }

        var line = d3.line().x(d => x(d[0])).y(d => y(d[1]));
        
        g_objects.selectAll(".object_area")
            .data(self.data.scatter.sort((a, b) => a.object - b.object))
            .join("path")
            .attr("d", d => line(d.shape))
            .attr("fill", d => self.color_scale(d[self.color_attr]))
            .attr("mask", d => {
                if (d.style == "solid") {
                    return "None";
                } else {
                    console.log(`url(#${self.spurious_crossing_mark}_mask)`);
                    return `url(#${self.spurious_crossing_mark}_mask)`;
                }
            })
            //.attr("stroke", d => self.color_scale(d[self.color_attr]))
            //.attr("stroke-width", 2.5)
            //.attr("stroke-linejoin", "round")
            //.attr("stroke-opacity", 0.85)
            .attr("opacity", 1)
            .attr("class", d => "object_area object_area_deselected object_area_" + d.object)
            .attr("id", d => "object_area_" + d.object + "_" + d.timestep.toString().replace(".", "_"))
            //.style("mix-blend-mode", "hard-light")
            .on("click", click_handler)
            .on("mouseover", mouseover)
            .on("mousemove", mousemove)
            .on("mouseleave", mouseleave);
        
        function click_handler(){
            var selection = d3.select(this);
            var d = selection.data()[0];
            var object_id = d.object;
            console.log(object_id)
            console.log("entrou aqui")
            if (d3.select(this).classed("object_area_selected")) {
                console.log("entrou no selecionado")
                d3.selectAll(".object_area_" + object_id)
                    .attr("class", "object_area object_area_deselected object_area_" + object_id);
            } else {
                console.log("entrou na deselecionado")
                d3.selectAll(".object_area_" + object_id)
                    .attr("class", "object_area object_area_selected object_area_" + object_id);
            }
            
            if (d3.selectAll(".object_area_selected").empty()) {
                d3.selectAll(".object_area").attr("opacity", 1);
            } else {
                d3.selectAll(".object_area_selected").attr("opacity", 1);
                d3.selectAll(".object_area_deselected").attr("opacity", 0.25);
            }
            
            self.update_data_selection();
        };
    }

    update_data_selection(){
        const self = this;
        const type  = this.dataset_config_cur.type;

        const selected_objects = [... new Set(d3.selectAll(".object_area_selected")
            .data()
            .map(d => d.object))];
        this.interactive_selections.object_clicked = selected_objects;
        
        if (type == "video"){
            this.video_player.set_selected_objects(selected_objects);
        } else if (type == "map") {
            this.webgl_map.set_selected_objects(selected_objects);
        }
    }

    /**
     * Draw the colorbar with the data from this.data.colorbar.
     * @param {group} g 
     * @param {scale} y 
     */
    plot_colorbar = (g, y) => {
        const self = this;
        var clip = g.append("defs").append("svg:clipPath")
            .attr("id", "clip-colorbar")
            .append("svg:rect")
            .attr("id", "clip-rect")
            .attr("x", 0)
            .attr("y", 0)
            .attr("width", 20)
            .attr("height", this.height);

        var g_colorbar = g.append("g")
            .attr("clip-path", "url(#clip-colorbar)");

        const colorbar_rects = g_colorbar.selectAll("colorbar_rect")
            .data(this.data.scatter)
            .join("rect")
            .attr("width", 20)
            .attr("height", d => (y(d.y - d.height/2) - y(d.y + d.height/2)))
            .attr("x", 0)
            .attr("y", d => y(d.y + d.height/2))
            .attr("opacity", 1)
            .attr("fill", d => rgbToHex(...d["color_"+self.colormap]
                .slice(1, -1)
                .split(",")
                .map(d => parseInt(d))));
            
        g_colorbar.append("rect")
            .attr("width", 20)
            .attr("x", 0)
            .attr("y", 0)
            .attr("height", this.height)
            .attr("fill", "white")
            .attr("opacity", 0.001)
            .on("mouseover", function(){
                self.plot_colormap = true;
                if (self.dataset_config_cur.type == "video"){
                    self.video_player.set_plot_color_map(true);
                    self.video_player.render();
                } else if (self.dataset_config_cur.type == "map") {
                    self.webgl_map.map_texture = true;
                    self.webgl_map.render();
                }
            })
            .on("mouseleave", function(){
                self.plot_colormap = false;
                if (self.dataset_config_cur.type == "video"){
                    self.video_player.set_plot_color_map(false);
                    self.video_player.render();
                } else if (self.dataset_config_cur.type == "map") {
                    self.webgl_map.map_texture = false;
                    self.webgl_map.render();
                }
            });

        return colorbar_rects;
    }

    highlight_intersections(){
        const self = this;
        if (!self.highlight_intersections_active){
            // Highlight objects of the selected timestep
            // Change class of all objects to deselected
            d3.selectAll(".object_area")
                .attr("class", d => "object_area object_area_deselected object_area" + d.object)
                .attr("opacity", 0.1);
            self.update_data_selection();

            // Than change the opacity of the data of this spurious intersection 
            self.data.intersections.forEach(function(e) {
                console.log(e.timestep.toString().replace(".", "_"))    
                d3.selectAll("#object_area_" + e.object1 + "_" + e.timestep.toString().replace(".", "_"))
                .attr("opacity", 1);
                d3.selectAll("#object_area_" + e.object2 + "_" + e.timestep.toString().replace(".", "_"))
                .attr("opacity", 1);
            });
        } else{
            d3.selectAll(".object_area")
                .attr("opacity", 1);
        }
        self.highlight_intersections_active = !self.highlight_intersections_active;
    }

    /**
     * Plot procedure, call the individual functions.
     */
    plot(){
        const self = this;

        // Create groups to plot individual objects
        const center_plot = this.svg_plot
            .append("g")
            .attr("transform", 
            `translate(${this.margin.left + this.colorBar_width}, ${this.margin.top})`);
        const center_plot_area_error = center_plot.append("g");
        const center_plot_main = center_plot.append("g")
            .attr("transform", `translate(0, ${this.area_error_spurious_height})`);
        const center_plot_x_axis = center_plot.append("g")
            .attr("transform", `translate(0, ${this.area_error_spurious_height + this.height})`);
        const center_plot_errors_bounding_box =  center_plot.append("g")
            .attr("id", "g_errors_bounding_box")
            .attr("opacity", 0.5)
            .attr("transform", `translate(0, ${this.area_error_spurious_height})`);
        const left_plot = this.svg_plot.append("g")
            .attr("transform", 
            `translate(${this.margin.left}, ${this.margin.top + this.area_error_spurious_height})`);

        // Creating pattern to use later
        var defs = this.svg_plot.append("defs");
        
        defs
        .append('pattern')
        .attr('id', 'dashed_pattern')
        .attr("patternUnits", "objectBoundingBox")
        .attr("patternContentUnits", "objectBoundingBox")
        .attr("patternTransform", "rotate(45)")
        .attr("width", 1)
        .attr("height", 0.2)
        .append('rect')
        .attr("x", 0)
        .attr("y", 0)
        .attr("width", 1)
        .attr("height", 0.1)
        .attr('fill', '#000000');

        var mask = defs.append('mask')
            .attr('id', 'dashed_mask')
            .attr("maskUnits", "objectBoundingBox")
            .attr("maskContentUnits", "objectBoundingBox")
            .attr('width', 1)
            .attr('height', 1)
            .attr('x', 0)
            .attr('y', 0);
        
        mask.append("rect")
            .attr("x", 0)
            .attr("y", 0)
            .attr("width", 1)
            .attr("height", 1)
            .attr("fill", "white");

         mask.append('rect')
                .attr("x", 0)
                .attr("y", 0)
                .attr("width", 1)
                .attr("height", 1)
                .attr("fill", "url(#dashed_pattern)");

        defs
        .append("linearGradient") 
            .attr("id", "gradient_pattern")
            .attr("x1", 0)
            .attr("x2", 1)
            .attr("y1", 0)
            .attr("y2", 0)
            .html(`
                <stop offset="0%" style="stop-color: #fff; stop-opacity: 1.0"/>
                <stop offset="30%" style="stop-color: #fff; stop-opacity: 0.2"/>
                <stop offset="70%" style="stop-color: #fff; stop-opacity: 0.2"/>
                <stop offset="100%" style="stop-color: #fff; stop-opacity: 1.00"/>
            `);
        
        var mask = defs.append('mask')
            .attr('id', 'gradient_mask')
            .attr("maskUnits", "objectBoundingBox")
            .attr("maskContentUnits", "objectBoundingBox")
            .attr('width', 1)
            .attr('height', 1)
            .attr('x', 0)
            .attr('y', 0);
    
         mask.append('rect')
                .attr("x", 0)
                .attr("y", 0)
                .attr("width", 1)
                .attr("height", 1)
                .attr("fill", "url(#gradient_pattern)");

        // Creating tooltip
        d3.select("#intersection_plot")
            .append("div")
            .style("display", "none")
            .attr("class", "tooltip")
            .attr("id", "intersection_view_tooltip")
            .style("position", "absolute")
            .style("background-color", "white")
            .style("border", "solid")
            .style("border-width", "2px")
            .style("border-radius", "5px")
            .style("padding", "5px");
        
        d3.select("#morevis_plot")
            .append("div")
            .style("display", "none")
            .attr("class", "tooltip")
            .attr("id", "main_plot_tooltip")
            .style("position", "absolute")
            .style("background-color", "white")
            .style("border", "solid")
            .style("border-width", "2px")
            .style("border-radius", "5px")
            .style("padding", "5px");

        var spurious_plot_tooltip = d3.select("#morevis_plot")
            .append("div")
            .style("display", "none")
            .attr("class", "tooltip")
            .attr("id", "spurious_plot_tooltip")
            .style("position", "absolute")
            .style("background-color", "white")
            .style("border", "solid")
            .style("border-width", "1px")
            .style("padding", "5px")
         
        spurious_plot_tooltip
            .append("svg")
            .attr("width", "100%")
            .attr("height", "100%")
            .attr("id", "spurious_plot_tooltip_svg");
        
        spurious_plot_tooltip
            .append("div")
            .style("display", "none")
            .attr("class", "tooltip")
            .attr("id", "spurious_plot_tooltip_inside_tooltip")
            .style("position", "absolute")
            .style("background-color", "white")
            .style("border", "solid")
            .style("border-width", "2px")
            .style("border-radius", "5px")
            .style("padding", "5px");

        // Create scales 
        var x = d3.scaleLinear()
            .domain([
                d3.min(this.data.scatter.map(d => d.timestep)), 
                d3.max(this.data.scatter.map(d => d.timestep)) + 1
            ])
            .range([0, this.width]);
        
        var y = d3.scaleLinear()
            .domain([
                d3.min(this.data.scatter.map(d => d.y - d.height/2)), 
                d3.max(this.data.scatter.map(d => d.y + d.height/2))
            ])
            .range([this.height, 0]);
        
        var [bar_plot_intersections, bar_plot_spurious] = this.plot_error_spurious(
            center_plot_area_error, x
        );

        var colorbar_rects = this.plot_colorbar(left_plot, y);
        var xAxis = this.plot_x_axis(center_plot_x_axis, x);
        
        var line_grid = this.plot_grid(center_plot_main, x); 
        
        //console.log("plotou grid")
        var yAxis = this.plot_y_axis(left_plot, y);
        //console.log("plotou eixo y")
        this.plot_objects_shapes(center_plot_main, x, y);
        
        
        //plot color legend
        if(this.color_attr_info.type == 'continuous'){
            self.color_legend_svg.selectAll("*").remove();
    
            Legend(this.color_scale, self.color_legend_svg, {
                title: this.color_attr_info.label,
                width: 200,
                ticks: 200/8,
                tickFormat: d3.format(".0f")
            });
        }

        // Plot axis labels
        center_plot_area_error
            .append("text")
            .attr("x", -18)
            .attr("y", 28)
            .attr("fill", "black")
            .style("font-weight", 600)
            .text("1D Space");

        center_plot_x_axis
            .append("text")
            .attr("x", self.width/2)
            .attr("y", 25)
            .attr("fill", "black")
            .style("font-weight", 600)
            .text("Timesteps");

        // Creating zoom 
        const zoom = d3.zoom()
                //.translateExtent([[0, 0], [this.width, this.height]])
                .scaleExtent([1, 8])
                .on("zoom", zoomed);
        
        center_plot_main.call(zoom);

        // Creating brushes
        const g_brush = center_plot_main.append("g")
            .attr("class", "g_brush");
        
        var brush = d3.brush()
            .extent([[0, 0], [this.width, this.height]])
            .on("end", brushend);

        function brushend (event) {
            if (!event.selection) return;
            const x0 = Math.ceil(x.invert(event.selection[0][0]));
            const x1 = Math.floor(x.invert(event.selection[1][0]));
            const y0 = y.invert(event.selection[0][1]);
            const y1 = y.invert(event.selection[1][1]);

            var k = x0;
            self.selected_timesteps = [];
            while (k < x1) {
                self.selected_timesteps.push(k);
                k = k + 1;
            }
            self.selected_y_range = [Math.min(y0, y1), Math.max(y0, y1)];
            self.update_intersection_plot(x, y);
        }

        // Button to highlight intersections
        $("#highlight_intersections").on("click", function(e){
            e.preventDefault();
            self.highlight_intersections();
        })
        // Button to reset zoom
        this.zoom_active = true;
        $("#reset_zoom").on("click", function(e){
            e.preventDefault();
            if (!self.zoom_active) return;
            center_plot_main
                .transition()
                .duration(750)
                .call(zoom.transform, d3.zoomIdentity);
        });

        // Button to activate/deactivate zoom
        $("#activate_selection").on("click", function(e){
            e.preventDefault();
            if (self.zoom_active) {
                center_plot_main.on(".zoom", null);
                g_brush.call(brush);
                g_brush.selectAll("*")
                    .attr("pointer-events", "all");
            } else {
                g_brush.on(".brush", null);
                g_brush.selectAll("*")
                    .attr("cursor", "default")
                    .attr("pointer-events", "none");
                center_plot_main.call(zoom);
            }

            document.getElementById("activate_selection").innerHTML = (
                `${self.zoom_active ? "Deactivate" : "Activate"} Selection`
            );
            self.zoom_active = !self.zoom_active;
        });

        function zoomed(event) {
            // Update scales  
            var brush_selection = d3.brushSelection(d3.select(".g_brush").node());
            if (brush_selection) {
                var brush_x0 = x.invert(brush_selection[0][0]);
                var brush_x1 = x.invert(brush_selection[1][0]);
                var brush_y0 = y.invert(brush_selection[0][1]);
                var brush_y1 = y.invert(brush_selection[1][1]);
            }

            x.range([0, self.width]
                .map(d => event.transform.applyX(d)));
            y.range([self.height, 0]
                .map(d => event.transform.applyY(d)));            
            
            if (brush_selection) {
                d3.select(".g_brush")
                    .call(brush.move, [
                    [x(brush_x0), y(brush_y0)],
                    [x(brush_x1), y(brush_y1)]
                    ]);
            } 

            // Update plots

            // Update x-axis
            self.svg_plot.selectAll(".x-axis").call(xAxis);

            // Update y-axis
            self.svg_plot.selectAll(".y-axis").call(yAxis);

            // Update grid
            d3.selectAll(".grid-divisor")
                .attr("d", d => line_grid(x,d));
            
            // Update spurious error area chart
            bar_plot_intersections
                .attr("x", d => x(d.timestep))
                .attr("width", x(1) - x(0));

            bar_plot_spurious
                .attr("x", d => x(d.timestep))
                .attr("width", x(1) - x(0));

            // Update colorbar
            colorbar_rects
                .attr("height", d => (y(d.y - d.height/2) - y(d.y + d.height/2)))
                .attr("y", d => y(d.y + d.height/2));
            
            // Update object shapes
            var line = d3.line()
                .x(d => x(d[0]))
                .y(d => y(d[1]));
            d3.selectAll(".object_area")
                .attr("d", d => line(d.shape));
        }
    }
}


/*
Page interactivity
*/
$(document).ready(function(){
    var vis = new Visualization();

    $("#sidebar_close").on("click", function(e) {
        e.preventDefault();
        document.getElementById("sidebar").style.display = "none";
    });
    
    $("#sidebar_open").on("click", function(e) {
        e.preventDefault();
        document.getElementById("sidebar").style.display = "block";
    });

    
    $("#dataset").on("click", function(e) {
        e.preventDefault();
        vis.update_curve_color_selector();
    });

    $("#update").on("click", function(e){
        e.preventDefault();
        vis.update_handler();
    });
})




