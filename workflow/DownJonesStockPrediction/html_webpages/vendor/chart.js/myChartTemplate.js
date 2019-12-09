
function float2dollar(value){
    return (value).toFixed(4).replace(/\d(?=(\d{3})+\.)/g, '$&,');
}

function getaverage(dir)
{
	return "UNIQUEAV";
}
function getopen(dir)
{
	return "UNIQUEOPEN";
}
function gethigh(dir)
{
	return "UNIQUEHIGH";
}
function getlow(dir)
{
	return "UNIQUELOW";
}
function getclose(dir)
{
	return "UNIQUECLOSE";
}
function getvolume(dir)
{
	return "UNIQUEVOLUME"
}
function getadjclose(dir)
{
	return "UNIQUEADJ"
}
function getlabel(dir)
{
	return "UNIQUELABEL";
}

function renderChart(data, labels, chartname) {
	var bgcolor = [];
    var ctx = document.getElementById(chartname).getContext('2d');
    var myChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
				label: '',
                data: data,
                pointborderColor: bgcolor,
				pointBackgroundColor: bgcolor,
				borderColor: bgcolor,
                backgroundColor: 'rgba(25, 198, 225, 0.1)'
            }, {label: '50 Days stock price',backgroundColor: '#000080'}, 
			{label: 'Prediction Value', backgroundColor: '#FF0000'}]
        },
        options: { 
			legend: {
				labels: {
					filter: function(legendItem, chartData) {
						if (legendItem.datasetIndex === 0) {
							return false;
						}
							return true;
						}
					}
				},
			responsive: true,		
            scales: {
                yAxes: [{
                    ticks: {
                        beginAtZero: true,
                        callback: function(value, index, values) {
                            return float2dollar(value);
                        }
                    }
                }]                
            }			
        },
    });
	
	for (i = 0; i < myChart.data.datasets[0].data.length; i++) {
		if (i > 49) 
		{
			bgcolor.push('#FF0000');
		} 
		else 
		{
			bgcolor.push('#000080');
		}
	}
	myChart.update();
}

$(document).ready(
    function () {
        average = getaverage();
		opening = getopen();
		highest = gethigh();
		lowest = getlow();
		closing = getclose();
		adj = getadjclose();
		volume = getvolume();
        labels =  getlabel();
        renderChart(average, labels, "average");
		renderChart(opening, labels, "open");
		renderChart(highest, labels, "high");
		renderChart(lowest, labels, "low");
		renderChart(closing, labels, "close");
		renderChart(volume, labels, "volume");
		renderChart(adj, labels, "adjusted close");
    }
);