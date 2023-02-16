class Network{
    constructor(
        data,
        timestep,
        color_scale, 
        link_scale, 
        color_attr_mapper, 
        width, 
        height, 
        g,
        tooltip,
        optim_pos) {
        this.data = data;
        this.timestep = timestep;
        this.color_scale = color_scale;
        this.link_scale = link_scale;
        this.color_attr_mapper = color_attr_mapper;
        this.g = g;
        this.tooltip = tooltip;
        this.optim_pos = optim_pos;
        this.radius = 4;
        this.width = width;
        this.height = height;
        return;
    }

    plot = () => {
        const self = this;
        const data = this.data;

        //  Draw border and info text
        this.g
            .append("rect")
            .attr("width", self.width)
            .attr("height", self.height)
            .attr("x", 0)
            .attr("y", 0)
            .attr("fill", "none")
            .attr("stroke", "#eee")
            .attr("stroke-width", 1)

        this.g
            .append("text")
            .attr("x", 5)
            .attr("y", 15)
            .attr("fill", "#000")
            .text("t = " + self.timestep);
            
        const link = this.g
            .selectAll("line")
            .data(data.links)
            .join("line")
            .attr("stroke-width", d => d.spurious == 0 ? self.link_scale(d.value) : 1.5)
            .style("stroke", d => d.spurious == 0 ? "#aaa" : "red");

        const node = this.g
            .selectAll("circle")
            .data(data.nodes)
            .join("circle")
                .attr("r", self.radius)
                .style("fill", d => self.color_scale(self.color_attr_mapper(d)))  

        const simulation = d3.forceSimulation(data.nodes)                
            .force("link", d3.forceLink()                             
                .id(function(d) { return d.id; })                    
                .links(data.links)                                   
            )
            .force("charge", d3.forceManyBody().strength(-400))        
            .force("center", d3.forceCenter(this.width / 2, this.height / 2).strength(0.5))    
            .force("bounds", boxingForce)
            .stop();

            

        // Custom force to put all nodes in a box
        function boxingForce() {
            const radius = self.radius;
            const margin = 0.2;
            for (let node of data.nodes) {
                // Of the positions exceed the box, set them to the boundary position.
                // You may want to include your nodes width to not overlap with the box.
                node.x = Math.max(self.width*margin-radius, Math.min(self.width*(1 - margin) - radius, node.x));
                node.y = Math.max(self.height*margin-radius, Math.min(self.height*(1 - margin) - radius, node.y));
            }
        }
        
        if (this.optim_pos) {
            for(let i = 0; i < 300; i++) simulation.tick();
        }

        function ticked() {
            node
            .attr("cx", function(d) {
                return d.x = Math.max(self.radius, Math.min(self.width - self.radius, d.x)); 
            })
            .attr("cy", function(d) {
                return d.y = Math.max(self.radius, Math.min(self.height - self.radius, d.y)); 
            });

            link
                .attr("x1", function(d) { return d.source.x; })
                .attr("y1", function(d) { return d.source.y; })
                .attr("x2", function(d) { return d.target.x; })
                .attr("y2", function(d) { return d.target.y; });
        }

        ticked();
    }

    plot_vertical = () => {
        const self = this;
        const data = this.data;

        //  Create scales
        const node_scale = d3.scaleBand()
            .domain(data.nodes.map(d => d.id))
            .range([0, self.height - 25]);

        const link_scale = d3.scaleBand()
            .domain(data.links.map(d => d.source + "," + d.target))
            .range([20, self.width - 20]);


        //  Draw background
        const main_g = this.g.append("g")
            .attr("transform", `translate(10, 25)`);    

        main_g.selectAll(".node_rects")
            .data(data.nodes)
            .enter()
            .append("rect")
            .attr("x", -10)
            .attr("y", d => node_scale(d.id) - node_scale.bandwidth()/2)
            .attr("width", self.width)
            .attr("height", node_scale.bandwidth())
            .attr("fill", (d, i) => i % 2 == 0 ? "#fff" : "#eee");


        //  Draw border and info text
        this.g
            .append("rect")
            .attr("width", self.width)
            .attr("height", self.height)
            .attr("x", 0)
            .attr("y", 0)
            .attr("fill", "none")
            .attr("stroke", "#ccc")
            .attr("stroke-width", 1);

        this.g
            .append("text")
            .attr("x", 5)
            .attr("y", 15)
            .attr("fill", "black")
            .text("t = " + self.timestep);

        // Three function that change the tooltip 
        // when user hover / move / leave a cell
        var mouseover = function(d) {
            self.tooltip.style("display", "block");
            d3.select(this)
                .attr("stroke", "black");
        }

        var mousemove_node = function(event, d) {
            self.tooltip
                .style("left", event.layerX + "px")
                .style("top",  event.layerY + "px")
                .html(`
                Object: (${d.id})<br>
                `);
        }

        var mousemove_link = function(event, d) {
            self.tooltip
                .style("left", event.layerX + "px")
                .style("top",  event.layerY + "px")
                .html(`
                Objects: (${d.source}, ${d.target})<br>
                Value: ${Math.round(d.value * 100)/100}<br>
                Spurious: ${d.spurious == 1}
                `);
        }

        var mouseleave_node = function(d) {
            self.tooltip.style("display", "none")
            d3.select(this)
                .attr("stroke", "none");
        }

        var mouseleave_link = function(d) {
            self.tooltip.style("display", "none")
            d3.select(this)
                .attr("stroke", d => d.spurious == 0 ? "#aaa" : "red");
        }

        // Draw nodes and connections

        main_g.selectAll(".node_mark")
            .data(data.nodes)
            .enter()
            .append("circle")
            .attr("cx", 0)
            .attr("cy", d => node_scale(d.id))
            .attr("r", 4)
            .attr("fill", d => this.color_scale(this.color_attr_mapper(d)))
            .on("mouseover", mouseover)
            .on("mousemove", mousemove_node)
            .on("mouseleave", mouseleave_node);

        main_g.selectAll(".link_mark")
            .data(data.links)
            .enter()
            .append("path")
            .attr("d", d => {
                var x = link_scale(d.source + "," + d.target);
                var y0 = node_scale(d.source);
                var y1 = node_scale(d.target);
                return d3.line()([[x, y0], [x, y1]]);
            })
            .attr("stroke-width", d => d.spurious == 0 ? self.link_scale(d.value) : 1.5)
            .attr("stroke", d => d.spurious == 0 ? "#aaa" : "red")
            .on("mouseover", mouseover)
            .on("mousemove", mousemove_link)
            .on("mouseleave", mouseleave_link);

        main_g.selectAll(".link_point")
            .data(data.links)
            .enter()
            .append("circle")
            .attr("r", 3)
            .attr("fill", "#606060")
            .attr("cx", d => link_scale(d.source + "," + d.target))
            .attr("cy", d =>  node_scale(d.source));

        main_g.selectAll(".link_point")
            .data(data.links)
            .enter()
            .append("circle")
            .attr("r", 3)
            .attr("fill", "#606060")
            .attr("cx", d => link_scale(d.source + "," + d.target))
            .attr("cy", d =>  node_scale(d.target));
    }
}