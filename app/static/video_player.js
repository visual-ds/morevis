class VideoPlayer{
    constructor(
        width,
        height,
        data,
        xrange,
        yrange,
        frame_map,
        image_path_map,
        color_map,
        color_map_name,
        )
    {   
        this.width = width;
        this.height = height;
        this.data = data;
        this.selected_objects = [];
        this.xrange = xrange;
        this.yrange = yrange;
        this.frame_map = frame_map;
        this.image_path_map = image_path_map;
        this.color_map = color_map;
        this.color_map_name = color_map_name;
        this.plot_color_map = false;
        this.start();
    }

    /**
     * Starts the video plot.
     */
    start(){
        const self = this;
        const min_frame = self.frame_map(d3.min(self.data.map(d => d.timestep)));
        const max_frame = self.frame_map(d3.max(self.data.map(d => d.timestep)));
        self.min_frame = min_frame;
        self.max_frame = max_frame;
        const step_frame = 2 * (d3.min(self.data.map(d => self.frame_map(d.timestep)).filter(d => d > min_frame)) - min_frame);

        document.getElementById("data_plot").innerHTML = "";
        document.getElementById("data_plot").innerHTML = `
        <div id = "video_player">
            <canvas id="video_image" width = "340" height = "253"></canvas>
        </div>
        <div id = "video_controller">
            <span id="video_timer">0</span><span style="font-size: 12px; width: 32px;">/${max_frame}</span>
            <button type="button" id="previous_frame"> << </button>
            <input type="range" min="${min_frame}" max="${max_frame}" step="${step_frame}" value="0" id="video_slide" style="width:200px; margin-top: 3px;">
            <button type="button" id="next_frame"> >> </button> 
        </div>
        `
        self.cur_frame = min_frame;
        self.render();
        //document.getElementById("data_plot_block_title").innerHTML = "VIDEO PLAYER";

        $("#video_slide").on("input", function(e){
            e.preventDefault();
            self.cur_frame = parseInt(Math.floor($("#video_slide").val()));
            self.render();
        });

        $("#previous_frame").on("click", function(e){
            e.preventDefault();
            if(self.cur_frame == self.min_frame) return;
            self.cur_frame = self.cur_frame - step_frame;
            $("#video_slide").val(self.cur_frame);  
            self.render();
        });

        $("#next_frame").on("click", function(e){
            e.preventDefault();
            if(self.cur_frame == self.max_frame) return;
            self.cur_frame = self.cur_frame + step_frame;
            $("#video_slide").val(self.cur_frame); 
            self.render();
        });
    }

    update_data(data){
        this.data = data;
        this.start();
    }

    set_cur_frame(cur_frame){
        this.cur_frame = cur_frame;
        $("#video_slide").val(this.cur_frame); 
        this.render();
    }

    set_plot_color_map(plot_color_map){
        this.plot_color_map = plot_color_map;
        this.render();
    }

    set_selected_objects(selected_objects){
        this.selected_objects = selected_objects;
        this.render();
    }
    
    render(){
        const self = this;
        document.getElementById("video_timer").innerHTML = self.cur_frame;
        
        var img = new Image;
        const width_ratio = self.width/self.xrange;
        const height_ratio = self.height/self.yrange;
        const canvas = document.getElementById("video_image");
        const ctx = canvas.getContext("2d");
        
        if(self.plot_color_map){
            ctx.globalAlpha = 0.5;
        } else {
            ctx.globalAlpha = 1;
        }
        
        img.onload = function(){
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0);

            //Temporal selection drawing bouding boxes
            self.data
                .filter(d => d.frame == self.cur_frame)
                .forEach(d => {
                    ctx.beginPath();
                    ctx.lineWidth = 4;
                    ctx.strokeStyle = self.color_map(d);
                    ctx.rect(d.xmin*width_ratio + 2, d.ymin*height_ratio + 2, 
                            (d.xmax - d.xmin)*width_ratio - 4, (d.ymax - d.ymin)*height_ratio - 4);
                    ctx.stroke();
                });
            
            if(self.plot_color_map){
                var colormap_img = new Image;
                colormap_img.onload = function(){
                    ctx.scale(1, -1);
                    ctx.drawImage(colormap_img, 0, 0, canvas.width, -canvas.height);
                    ctx.scale(1, -1);
                }	
                colormap_img.src = `http://127.0.0.1:5000/static/data/${self.color_map_name}.png`;
            }

            //object click selection drawing objec trace
            self.selected_objects.forEach(d => self.draw_object_trace(d));       
        }
        img.src = "http://127.0.0.1:5000/" + self.image_path_map(self.cur_frame);
    }

    /**
    * Draw the trace overtime of the object.
    * @param {Int} object 
    */ 
    draw_object_trace(object){
        const self = this;

        const trace_data = this.data
            .filter(d => d.object == object)
            .sort(function(a, b){ return a.frame - b.frame});
        
        const canvas = document.getElementById("video_image");
        const ctx = canvas.getContext("2d");
        const width_ratio = self.width/self.xrange;
        const height_ratio = self.height/self.yrange;

        var scale_width = d3.scaleLinear()
            .domain([0, trace_data.length])
            .range([2, 10]);
        
        for(let i = 1; i < trace_data.length; i++){
            var x0 = trace_data[i-1].xcenter;
            var y0 = trace_data[i-1].ycenter;                
            var x1 = trace_data[i].xcenter;
            var y1 = trace_data[i].ycenter; 
            ctx.beginPath();
            ctx.lineWidth = scale_width(i);
            ctx.strokeStyle = self.color_map(trace_data[i]);  
            ctx.moveTo(x0*width_ratio, y0*height_ratio);      
            ctx.lineTo(x1*width_ratio, y1*height_ratio);
            ctx.stroke();
        }
    }
}