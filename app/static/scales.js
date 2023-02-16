class distortionScale{
    constructor(){
      this.linear = d3.scaleLinear()
    }

    domain(data, acess_value, distortion){
        this.distortion = distortion;
        this.minVal = 0;//d3.min(data.map(d => acess_value(d)));
        this.maxVal = d3.max(data.map(d => acess_value(d)));
        var minVal = this.minVal;
        var maxVal = this.maxVal;
        this.n_steps = (maxVal - minVal);

        this.bins = [];
        for(let i = minVal; i <= maxVal; i++){
            this.bins.push({"x": i, "count": 0})
        }

        this.linear.domain([minVal, maxVal]) 
        
        for(let i = 0; i < data.length; i++){
            var x = acess_value(data[i]);
            var ind = parseInt(x - this.minVal);
            this.bins[ind].count++;
        }

        for(let i = 0; i < this.bins.length; i++){
            this.bins[i].proportion = distortion ? this.bins[i].count/data.length : 1/this.bins.length;
        }

        //console.log(this.bins)
    }
    
    range(rangeVal){
        this.linear.range(rangeVal)
        this.range_width = rangeVal[1] - rangeVal[0]
        var cur_start = rangeVal[0]
        var cur_end;
        for(let i = 0; i < this.bins.length; i++){
            var cur_end = cur_start + this.range_width * this.bins[i].proportion;
            this.bins[i].width = this.range_width * this.bins[i].proportion;
            this.bins[i].center = (cur_start + cur_end)/2;
            this.bins[i].distorted_left = cur_start;
            this.bins[i].distorted_right = cur_end;
            this.bins[i].linear_left = this.linear(this.bins[i].x - 0.5);
            this.bins[i].linear_right = this.linear(this.bins[i].x + 0.5);
            cur_start = cur_end;
        }
    }

    transform_linear(x){
        return this.linear(x);
    }
    
    transform(x){
        if(!this.distortion){
            return this.linear(x);
        }
        else if (x % 1 == 0){
            var ind = Math.min(parseInt(x - this.minVal), this.bins.length -1);
            //console.log(ind);
            return this.bins[ind].center;
        }else {
            var ind = Math.min(parseInt(Math.floor(x + 0.5 - this.minVal)), this.bins.length -1);
            //console.log(ind);
            return this.bins[ind].center;
        }
        
    }

    get_width(x){
        if(!this.distortion){
            return this.linear(1) - this.linear(0);
        }
        else if (x % 1 == 0){
            var ind = Math.min(parseInt(x - this.minVal), this.bins.length - 1);
            return this.bins[ind].proportion * this.range_width;
        }else {
            var ind = Math.min(parseInt(Math.floor(x + 0.5 - this.minVal)), this.bins.length -1);
            return this.bins[ind].proportion * this.range_width;
        }
    }

    invert(x){
        if (!this.distortion) {
            return this.linear.invert(x);
        } else {
            if (this.bins[0].distorted_left >= x) {
                return this.bins[0].x;
            } else if (this.bins[this.bins.length - 1].distorted_right < x) {
                return this.bins[this.bins.length - 1].x;
            } else {
                for (let i = 0; i < this.bins.length; i++) {
                    if ((this.bins[i].distorted_left <= x) & 
                        (this.bins[i].distorted_right > x)) {
                        return this.bins[i].x;
                    }
                }
            }
        }
    }
}

class breakScale{
    constructor(){
      this.linear = d3.scaleLinear();
      this.linear_no_breaks =  d3.scaleLinear();
      this.breaks = [];
    }

    domain(data, acess_pos, acess_length, compute_breaks){
        var ind = data.map((d, i) => i);
        ind = ind.sort(function(a, b){ return acess_pos(data[a]) - acess_pos(data[b]);});
        var start_pos, end_pos, space;
      
        for(let i = 0; i < (data.length - 1); i++){
            if(i == 0){
                end_pos = acess_pos(data[ind[i]]) + acess_length(data[ind[i]])/2;
            }else{
                end_pos = d3.max(ind.slice(0, i + 1).map(ii => acess_pos(data[ii]) + acess_length(data[ii])/2));
            }
            start_pos = d3.min(ind.slice(i+1, ind.length).map(d => {
                return acess_pos(data[d]) - acess_length(data[d])/2;
            }));
            space = start_pos - end_pos;
            if(space > 0){
                this.breaks.push({start: end_pos, end: start_pos, space: space});
            }
        }
        
        this.break_size = d3.scaleLinear()
            .domain([d3.min(this.breaks.map(d => d.space)), 
                    d3.max(this.breaks.map(d => d.space))])
            .range([2, 5]);

        if(!compute_breaks){
            this.breaks = [];
        }

        var minVal = d3.min(data.map(d => acess_pos(d) - acess_length(d)/2));
        var maxVal = d3.max(data.map(d => acess_pos(d) + acess_length(d)/2));
        this.linear.domain([minVal, maxVal]);
        this.linear_no_breaks.domain([minVal, maxVal - this.breaks.map(d => d.space).reduce((a, b) => a + b, 0)]);   
    }
    
    range(rangeVal){
        this.linear.range(rangeVal);
        this.linear_no_breaks.range(rangeVal);
    }

    transform(y){
        var y_old = y;
        for(let i = 0; i < this.breaks.length; i++){
            if(y_old >= this.breaks[i].end){
                y = y - this.breaks[i].space;
            }
        }
        return this.linear_no_breaks(y);
    }

    from_breaks_to_original(y){
        if(this.breaks.length == 0){
            return y;
        }
        var i = 0;
        while((y > this.breaks[i].start)){
            y = y + this.breaks[i].space;
            i = i + 1;
            if(i == this.breaks.length){
                break;
            }
        }
        return Math.round(y);
    }

    invert(y){
        var y_invert = this.linear_no_breaks.invert(y);
        return this.from_breaks_to_original(y_invert);

    }
}