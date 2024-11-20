const cvsIn = document.getElementById("inputimg");
const ctxIn = cvsIn.getContext("2d");
const divOut = document.getElementById("pred");
let svgGraph = null;
let mouselbtn = false;


window.onload = () => {
    ctxIn.fillStyle = "white";
    ctxIn.fillRect(0, 0, cvsIn.width, cvsIn.height);
    ctxIn.lineWidth = 7;
    ctxIn.lineCap = "round";
    initProbGraph();
}

function initProbGraph() {
    const dummyData = Array(36).fill(1); 
    const margin = { top: 10, right: 10, bottom: 10, left: 20 };
    const width = 250;
    const height = 196;

    const yScale = d3.scaleLinear()
        .domain([35, 0]) 
        .range([height, 0]);

    svgGraph = d3.select("#probGraph")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    svgGraph.append("g")
        .attr("class", "y axis")
        .call(d3.axisLeft(yScale));

    const barHeight = 20;
    svgGraph.selectAll("svg")
        .data(dummyData)
        .enter()
        .append("rect")
        .attr("y", (d, i) => (yScale(i) - barHeight / 2))
        .attr("height", barHeight)
        .style("fill", "green")
        .attr("x", 0)
        .attr("width", d => d * 2);
}

cvsIn.addEventListener("mousedown", e => {
    if (e.button === 0) {
        const rect = e.target.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        mouselbtn = true;
        ctxIn.beginPath();
        ctxIn.moveTo(x, y);
    } else if (e.button === 2) {
        onClear();
    }
});

cvsIn.addEventListener("mouseup", e => {
    if (e.button === 0) {
        mouselbtn = false;
        onRecognition();
    }
});

cvsIn.addEventListener("mousemove", e => {
    const rect = e.target.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    if (mouselbtn) {
        ctxIn.lineTo(x, y);
        ctxIn.stroke();
    }
});

cvsIn.addEventListener("touchstart", e => {
    if (e.targetTouches.length === 1) {
        const rect = e.target.getBoundingClientRect();
        const touch = e.targetTouches[0];
        const x = touch.clientX - rect.left;
        const y = touch.clientY - rect.top;
        ctxIn.beginPath();
        ctxIn.moveTo(x, y);
    }
});

cvsIn.addEventListener("touchmove", e => {
    if (e.targetTouches.length === 1) {
        const rect = e.target.getBoundingClientRect();
        const touch = e.targetTouches[0];
        const x = touch.clientX - rect.left;
        const y = touch.clientY - rect.top;
        ctxIn.lineTo(x, y);
        ctxIn.stroke();
        e.preventDefault();
    }
});

cvsIn.addEventListener("touchend", e => onRecognition());
cvsIn.addEventListener("contextmenu", e => e.preventDefault());

document.getElementById("clearbtn").onclick = onClear;
function onClear() {
    mouselbtn = false;
    ctxIn.fillStyle = "white";
    ctxIn.fillRect(0, 0, cvsIn.width, cvsIn.height);
    ctxIn.fillStyle = "black";
}

function onRecognition() {
    console.time("time");

    cvsIn.toBlob(async blob => {
        const body = new FormData();
        body.append('img', blob, "dummy.png");
        try {
            const response = await fetch("./DigitRecognition", {
                method: "POST",
                body: body,
            });
            const resjson = await response.json();
            showResult(resjson);
        } catch (error) {
            alert("Error: " + error);
        }
    });

    console.timeEnd("time");
}

function showResult(res) {
    divOut.textContent = res.pred; 
    const charCode = res.pred.charCodeAt(0);
    let index;

    if (res.pred >= '0' && res.pred <= '9') {
        index = charCode - '0'.charCodeAt(0);  // For digits 0-9
    } else if (res.pred >= 'A' && res.pred <= 'Z') {
        index = charCode - 'A'.charCodeAt(0) + 10;  // For letters A-Z
    }

    // Update probability display
    document.getElementById("prob").innerHTML =
        "Probability: " + res.probs[index].toFixed(2) + "%"; 

    // Update the probability graph
    svgGraph.selectAll("rect")
        .data(res.probs)
        .transition()
        .duration(300)
        .style("fill", (d, i) => i === index ? "blue" : "green")
        .attr("width", d => d * 2);
}
